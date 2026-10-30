"""MPS correctness patch for torchaudio's WavLM (torch 2.8.0 / torchaudio 2.8.0).

THE BUG
-------
On the MPS backend the *fused* normalisation backward kernels return garbage
affine gradients (``weight.grad`` / ``bias.grad``) whenever the **input tensor
does not require grad**.  The wrong values are not even deterministic: two
identical backward passes give ``weight.grad`` that differ by ~1.5 absolute,
which is the signature of uninitialised device memory.  The gradient with
respect to the *input* is correct, and everything is correct as soon as the
input does require grad.

Measured with torch 2.8.0, ``nn.LayerNorm(512)`` on a (3, 249, 512) input,
upstream gradient built on the CPU RNG and copied to the device:

    input.requires_grad=False   grad_weight relative error 4.71, cosine  0.066
    input.requires_grad=True    grad_weight relative error 1.3e-06, cosine 1.000

An op sweep (diag_opsweep.json) shows the bug is confined to ``LayerNorm`` and
``BatchNorm``.  ``GroupNorm``, ``Linear``, ``Conv1d``, ``Embedding`` and
``MultiheadAttention`` are all correct in both cases.

What the kernel actually does: it never writes ``grad_weight`` / ``grad_bias``
at all and hands back the stale contents of the recycled MPS allocation.  Three
independent observations pin this down (diag_layernorm.json, run in call order):

  * the first (1, 249, 512) case returns exactly 0 -- a freshly zeroed buffer;
  * the next (3, 249, 512) case returns mean |grad| 0.0001261913, which is
    bit-for-bit the *previous* case's correct result, not its own;
  * after one patched backward has written the right answer into that buffer,
    ``revert()`` and re-run and the unpatched path "passes" with the identical
    value, reproducibly, in a fresh process.

This is also why training saw a *constant* 134.08: the same recycled buffer with
the same stale contents every update.

WHY IT DESTROYS WavLM FINE-TUNING
---------------------------------
``WavLMRed`` freezes the convolutional feature extractor and runs it under
``torch.no_grad()``.  The tensor that reaches
``encoder.feature_projection.layer_norm`` therefore has
``requires_grad == False`` while that LayerNorm's own weight and bias are still
being trained -- exactly the broken case.  Its 512-element ``weight.grad``
comes back with mean magnitude ~2.9 instead of ~1.1e-05 (a factor of 2.6e5) and
on its own contributes a gradient norm of ~134, which is why
our MPS timing log reports ``gradient_norm`` frozen at
134.078 / 134.081 / 134.094 / 134.092 while the CPU run varies normally
(7.066 -> 3.889).  The optimiser then takes a huge, data-independent step on
that LayerNorm every update and the encoder never learns.

The forward pass is NOT affected: CPU and MPS agree to ~9e-04 relative at the
final layer, both for the single-clip path and for the padded ``encode_batch``
path (diag_layers.json / diag_batch.json).

THE FIX
-------
``apply()`` replaces ``nn.LayerNorm.forward`` (and optionally the BatchNorm
forwards) so that, in the broken case only, the affine transform is applied
*outside* the fused kernel::

    F.layer_norm(x, shape, None, None, eps) * weight + bias

The fused kernel is then called without affine parameters, so it has no
``grad_weight`` to get wrong, and the scale/shift gradients flow through plain
``mul`` / ``add``, which MPS gets right.  Every other case -- CPU tensors,
inference / ``no_grad`` / ``inference_mode``, frozen norms, and inputs that do
require grad -- keeps the original fast fused path untouched.

USAGE
-----
    import wavlm_mps_patch
    wavlm_mps_patch.apply()           # before the model is built or used

train_s1.py, train_c2.py and mac_train.py call apply() before they build the model.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ['apply', 'is_applied', 'revert', 'verify', 'STATE']

STATE: dict = {'applied': False, 'original_layer_norm_forward': None,
               'original_batch_norm_forward': None, 'hits': 0}


def _needs_workaround(input_tensor, weight, bias):
    """True only for the exact broken MPS case."""
    if input_tensor.device.type != 'mps':
        return False
    if not torch.is_grad_enabled():
        return False
    if input_tensor.requires_grad:
        return False
    return ((weight is not None and weight.requires_grad)
            or (bias is not None and bias.requires_grad))


def _patched_layer_norm_forward(self, input):
    if _needs_workaround(input, self.weight, self.bias):
        STATE['hits'] += 1
        out = F.layer_norm(input, self.normalized_shape, None, None, self.eps)
        if self.weight is not None:
            out = out * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out
    return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)


def _patched_batch_norm_forward(self, input):
    """Same trick for BatchNorm (WavLM does not use it; included for safety)."""
    if _needs_workaround(input, self.weight, self.bias):
        STATE['hits'] += 1
        original_weight, original_bias = self.weight, self.bias
        try:
            self.weight, self.bias = None, None
            out = STATE['original_batch_norm_forward'](self, input)
        finally:
            self.weight, self.bias = original_weight, original_bias
        shape = [1, -1] + [1] * (input.dim() - 2)
        if original_weight is not None:
            out = out * original_weight.view(shape)
        if original_bias is not None:
            out = out + original_bias.view(shape)
        return out
    return STATE['original_batch_norm_forward'](self, input)


def apply(layer_norm=True, batch_norm=True):
    """Install the monkeypatches. Idempotent; returns a description dict."""
    if STATE['applied']:
        return {'applied': True, 'already': True}
    if layer_norm:
        STATE['original_layer_norm_forward'] = nn.LayerNorm.forward
        nn.LayerNorm.forward = _patched_layer_norm_forward
    if batch_norm:
        STATE['original_batch_norm_forward'] = nn.modules.batchnorm._BatchNorm.forward
        nn.modules.batchnorm._BatchNorm.forward = _patched_batch_norm_forward
    STATE['applied'] = True
    return {'applied': True, 'layer_norm': bool(layer_norm), 'batch_norm': bool(batch_norm),
            'torch': torch.__version__,
            'reason': 'MPS fused LayerNorm/BatchNorm backward returns garbage affine '
                      'gradients when input.requires_grad is False'}


def revert():
    if not STATE['applied']:
        return {'applied': False}
    if STATE['original_layer_norm_forward'] is not None:
        nn.LayerNorm.forward = STATE['original_layer_norm_forward']
        STATE['original_layer_norm_forward'] = None
    if STATE['original_batch_norm_forward'] is not None:
        nn.modules.batchnorm._BatchNorm.forward = STATE['original_batch_norm_forward']
        STATE['original_batch_norm_forward'] = None
    STATE['applied'] = False
    return {'applied': False}


def is_applied():
    return bool(STATE['applied'])


def verify(shape=(3, 249, 512), tolerance=1e-3, seed=0):
    """Self-check: LayerNorm affine gradients on MPS must match CPU.

    Returns a dict; ``ok`` is True when the patched MPS gradients agree with the
    CPU gradients to ``tolerance`` relative, in the frozen-input case.

    Caveat: because the broken kernel returns recycled memory, ``verify()`` can
    report ``ok`` on an UNPATCHED process once a correct value has been left in
    that buffer.  Only the first call in a fresh process reliably detects the
    bug, so treat ``ok=True`` as "the patch works", never as "the patch is
    unnecessary".
    """
    if not torch.backends.mps.is_available():
        return {'ok': False, 'reason': 'mps_unavailable'}

    def one(device, requires_grad):
        torch.manual_seed(seed)
        x = torch.randn(*shape)
        norm = nn.LayerNorm(shape[-1])
        with torch.no_grad():
            norm.weight.copy_(torch.randn(shape[-1]) * .1 + 1.)
            norm.bias.copy_(torch.randn(shape[-1]) * .1)
        torch.manual_seed(seed + 1)
        gout = torch.randn(*shape) * 1e-5          # built on the CPU RNG on purpose
        norm = norm.to(device)
        xd = x.to(device).requires_grad_(requires_grad)
        y = norm(xd)
        y.backward(gout.to(device))
        return (y.detach().to('cpu'), norm.weight.grad.detach().to('cpu'),
                norm.bias.grad.detach().to('cpu'))

    def relative(a, b):
        a, b = a.reshape(-1), b.reshape(-1)
        return float((a - b).abs().max() / a.abs().mean().clamp_min(1e-12))

    out = {}
    for requires_grad in (False, True):
        cpu = one('cpu', requires_grad)
        mps = one('mps', requires_grad)
        out[f'input_requires_grad={requires_grad}'] = {
            'forward_relative': relative(cpu[0], mps[0]),
            'grad_weight_relative': relative(cpu[1], mps[1]),
            'grad_bias_relative': relative(cpu[2], mps[2])}
    worst = max(v['grad_weight_relative'] for v in out.values())
    worst = max(worst, max(v['grad_bias_relative'] for v in out.values()))
    return {'ok': worst <= tolerance, 'patch_applied': STATE['applied'],
            'tolerance': tolerance, 'worst_relative': worst, 'cases': out,
            'workaround_hits': STATE['hits']}


if __name__ == '__main__':
    import json
    torch.set_num_threads(3)
    before = verify()
    apply()
    after = verify()
    print(json.dumps({'before_patch': before, 'after_patch': after}, indent=2))
