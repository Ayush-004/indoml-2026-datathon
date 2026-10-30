# Copyright (c) Alibaba, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (see ../LICENSE).
#
# DERIVED 2026-09-30 for IndoML 2026 Track 2 from three upstream files of
# https://github.com/modelscope/modelscope at commit
# 6cb35d56b5e45cf30154d85b56db3c67edb032ac:
#   modelscope/pipelines/audio/ans_pipeline.py   (ANSZipEnhancerPipeline.preprocess/forward)
#   modelscope/utils/audio/audio_utils.py         (audio_norm, copied verbatim)
#   modelscope/models/audio/ans/zipenhancer.py    (ZipenhancerDecorator.forward)
# The unmodified upstream files are in ../upstream_unmodified/.
#
# Changes against upstream:
#   * plain functions, no ModelScope Pipeline class, no print(), no file I/O;
#   * float output (the caller converts to int16; upstream does
#     (y * 32768).astype(np.int16) at the audio_norm level);
#   * a batch of B > 1 rows gets one norm factor per row (upstream sums over the
#     whole batch; the pipeline itself only uses B = 1, where the code is the same);
#   * rows with zero energy give zeros (upstream gives NaN);
#   * optional length fixes (fix_length=True):
#       (a) whole-clip path (2 s <= T <= 6 s): torch.istft gets length=T.
#           Upstream calls istft without length and returns 100 * floor(T / 100)
#           samples, i.e. up to 99 samples fewer than the input.
#       (b) segment path (T > 6 s): the last window also writes its right edge.
#           Upstream never writes the last 4000 samples of the padded signal, so
#           real samples stay at zero when the zero padding is < 4000 samples.
#     With fix_length=False the functions return the upstream result.
#   * optional attention batch split (set_attention_budget) and convolution
#     frequency split (set_conv_budget): the same numbers (up to float
#     rounding) with less peak memory. They change the call, not the model code.
#   * forward_batch: a differentiable torch version of the same path for (B, T).

import numpy as np
import torch
import torch.nn.functional as F

from .zipenhancer import mag_pha_stft
from .zipenhancer_layers.zipformer import Zipformer2EncoderLayer

SAMPLE_RATE = 16000
N_FFT = 400
HOP_SIZE = 100
WIN_SIZE = 400
COMPRESS_FACTOR = 0.3
WINDOW = 16000 * 2  # 2 s windows in the segment path
STRIDE = int(WINDOW * 0.75)  # 1.5 s
GIVE_UP = (WINDOW - STRIDE) // 2  # 0.25 s dropped at each inner window edge
SEGMENT_ABOVE = WINDOW * 3  # T > 6 s -> segment path


def audio_norm(x):
    # verbatim copy of modelscope/utils/audio/audio_utils.py:audio_norm
    rms = (x**2).mean()**0.5
    scalar = 10**(-25 / 20) / rms
    x = x * scalar
    pow_x = x**2
    avg_pow_x = pow_x.mean()
    rmsx = pow_x[pow_x > avg_pow_x].mean()**0.5
    scalarx = 10**(-25 / 20) / rmsx
    x = x * scalarx
    return x


def mag_pha_istft_len(mag,
                      pha,
                      n_fft,
                      hop_size,
                      win_size,
                      compress_factor=1.0,
                      center=True,
                      length=None):
    # zipenhancer.mag_pha_istft plus the `length` argument of torch.istft.
    # length=None gives the upstream function exactly.
    mag = torch.pow(mag, (1.0 / compress_factor))
    com = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
    hann_window = torch.hann_window(win_size, device=com.device)
    wav = torch.istft(
        com,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        length=length)
    return wav


def enhance_rows(model, noisy_wav, fix_length=True):
    """ZipenhancerDecorator.forward for a (B, T) float tensor -> (B, T') tensor.

    model: the ZipEnhancer network (not the decorator).
    T' = T with fix_length=True, else 100 * floor(T / 100) as upstream.
    B = 1 uses the upstream expression for the norm factor unchanged.
    """
    t = noisy_wav.shape[1]
    if noisy_wav.shape[0] == 1:
        energy = torch.sum(noisy_wav**2.0)
        ok = bool(energy > 0)
        norm_factor = torch.sqrt(noisy_wav.shape[1] / energy) if ok else 1.0
        silent = None if ok else torch.ones(1, 1, dtype=torch.bool, device=noisy_wav.device)
    else:
        energy = torch.sum(noisy_wav**2.0, dim=1, keepdim=True)
        silent = energy <= 0
        norm_factor = torch.sqrt(
            t / torch.where(silent, torch.ones_like(energy), energy))
        if not bool(silent.any()):
            silent = None
    noisy_audio = (noisy_wav * norm_factor)

    mag, pha, com = mag_pha_stft(
        noisy_audio,
        N_FFT,
        HOP_SIZE,
        WIN_SIZE,
        compress_factor=COMPRESS_FACTOR,
        center=True)
    amp_g, pha_g, com_g, _, others = model(mag, pha)
    wav = mag_pha_istft_len(
        amp_g,
        pha_g,
        N_FFT,
        HOP_SIZE,
        WIN_SIZE,
        compress_factor=COMPRESS_FACTOR,
        center=True,
        length=t if fix_length else None)

    wav = wav / norm_factor
    if silent is not None:
        wav = torch.where(silent, torch.zeros_like(wav), wav)
    return wav


def upstream_padding(t):
    """(segment?, number of zeros appended) exactly as ANSZipEnhancerPipeline.forward."""
    window, stride = WINDOW, STRIDE
    segment = t > window * 3
    if t < window:
        return segment, window - t
    if segment:
        if t < window + stride:
            return segment, window + stride - t
        if (t - window) % stride != 0:
            return segment, ((t - window) // stride + 1) * stride + window - t
    return segment, 0


def upstream_output_info(t):
    """Length and zeroed tail of the upstream int16 output for an input of t samples."""
    segment, pad = upstream_padding(t)
    if segment:
        return {'path': 'segments_2s', 'pad': pad, 'out_len': t,
                'zeroed_tail': max(0, GIVE_UP - pad)}
    if t < WINDOW:
        return {'path': 'pad_to_2s', 'pad': pad, 'out_len': t, 'zeroed_tail': 0}
    return {'path': 'whole_clip', 'pad': 0,
            'out_len': HOP_SIZE * (t // HOP_SIZE), 'zeroed_tail': 0}


def pipeline_forward(model, ndarray, nsamples, device, fix_length=True):
    """ANSZipEnhancerPipeline.forward without the final int16 conversion.

    ndarray: (1, T) float32 numpy array after audio_norm (upstream preprocess).
    Returns a float64 numpy array: nsamples values (fix_length=True) or the
    upstream length and values (fix_length=False).
    """
    if isinstance(ndarray, torch.Tensor):
        ndarray = ndarray.cpu().numpy()
    decode_do_segement = False
    window = WINDOW
    stride = STRIDE
    b, t = ndarray.shape
    if t > window * 3:
        decode_do_segement = True
    if t < window:
        ndarray = np.concatenate(
            [ndarray, np.zeros((ndarray.shape[0], window - t))], 1)
    elif decode_do_segement:
        if t < window + stride:
            padding = window + stride - t
            ndarray = np.concatenate(
                [ndarray, np.zeros((ndarray.shape[0], padding))], 1)
        else:
            if (t - window) % stride != 0:
                padding = ((t - window) // stride + 1) * stride + window - t
                ndarray = np.concatenate(
                    [ndarray, np.zeros((ndarray.shape[0], padding))], 1)
    with torch.no_grad():
        ndarray = torch.from_numpy(np.float32(ndarray)).to(device)
        b, t = ndarray.shape
        if decode_do_segement:
            outputs = np.zeros(t)
            give_up_length = (window - stride) // 2
            current_idx = 0
            while current_idx + window <= t:
                tmp_output = enhance_rows(
                    model, ndarray[:, current_idx:current_idx + window],
                    fix_length=True)[0].cpu().numpy()
                end_index = current_idx + window - give_up_length
                if current_idx == 0:
                    outputs[current_idx:end_index] = tmp_output[:-give_up_length]
                else:
                    outputs[current_idx + give_up_length:end_index] = tmp_output[
                        give_up_length:-give_up_length]
                if fix_length and current_idx + window + stride > t:
                    # (b) last window: keep its right edge (upstream: zeros)
                    outputs[end_index:current_idx + window] = tmp_output[
                        -give_up_length:]
                current_idx += stride
        else:
            outputs = enhance_rows(
                model, ndarray, fix_length=fix_length)[0].cpu().numpy()
    return np.asarray(outputs[:nsamples], dtype=np.float64)


def whole_forward(model, ndarray, nsamples, device):
    """One pass over the whole clip for any length (not upstream for T > 6 s).

    Clips shorter than 2 s are zero padded to 2 s as upstream. Use with
    set_attention_budget for long clips: the time attention grows with T^2.
    """
    if isinstance(ndarray, torch.Tensor):
        ndarray = ndarray.cpu().numpy()
    t = ndarray.shape[1]
    if t < WINDOW:
        ndarray = np.concatenate(
            [ndarray, np.zeros((ndarray.shape[0], WINDOW - t))], 1)
    with torch.no_grad():
        x = torch.from_numpy(np.float32(ndarray)).to(device)
        y = enhance_rows(model, x, fix_length=True)[0].cpu().numpy()
    return np.asarray(y[:nsamples], dtype=np.float64)


def forward_batch(model, x, fix_length=True, mode='official', window_batch=None):
    """Differentiable torch version of pipeline_forward for a (B, T) batch.

    Each row is handled like a separate upstream call (own norm factor, own
    segmentation). The input level does not matter (the model normalises each
    window), so audio_norm is not needed. Returns (B, T); with
    fix_length=False the upstream short or zero tails are zero padded to T.
    window_batch: in the segment path, give the network at most this many
    2 s windows per call (None = all B * n_windows at once).
    The model must be in eval() mode: the vendored ActivationDropoutAndLinear
    has no train-mode forward (upstream returns None there).
    """
    bsz, t = x.shape
    if t < WINDOW:
        y = enhance_rows(model, F.pad(x, (0, WINDOW - t)), fix_length=True)
        return y[:, :t]
    if mode == 'whole' or t <= SEGMENT_ABOVE:
        y = enhance_rows(model, x, fix_length=fix_length)
        if y.shape[1] < t:
            y = F.pad(y, (0, t - y.shape[1]))
        return y[:, :t]
    _, pad = upstream_padding(t)
    xp = F.pad(x, (0, pad))
    tp = t + pad
    starts = list(range(0, tp - WINDOW + 1, STRIDE))
    segs = torch.stack([xp[:, s:s + WINDOW] for s in starts],
                       dim=1).reshape(bsz * len(starts), WINDOW)
    step = segs.shape[0] if not window_batch else int(window_batch)
    ys = torch.cat([
        enhance_rows(model, segs[i:i + step], fix_length=True)
        for i in range(0, segs.shape[0], step)
    ], dim=0).reshape(bsz, len(starts), WINDOW)
    pieces = []
    for k in range(len(starts)):
        a = 0 if k == 0 else GIVE_UP
        last = k == len(starts) - 1
        e = WINDOW if (last and fix_length) else WINDOW - GIVE_UP
        pieces.append(ys[:, k, a:e])
    y = torch.cat(pieces, dim=1)
    if y.shape[1] < t:
        y = F.pad(y, (0, t - y.shape[1]))
    return y[:, :t]


def _chunked_forward(layer, budget_bytes):
    base = type(layer).forward
    heads = layer.self_attn_weights.num_heads

    def forward(src,
                pos_emb,
                chunk_size=-1,
                attn_mask=None,
                src_key_padding_mask=None):
        seq_len, batch = src.shape[0], src.shape[1]
        # attention scores (S x S) + relative position scores (S x (2S-1)) per head
        per_row = heads * seq_len * (3 * seq_len - 1) * src.element_size()
        rows = max(1, budget_bytes // max(per_row, 1))
        if rows >= batch or (attn_mask is not None and attn_mask.dim() == 3):
            return base(layer, src, pos_emb, chunk_size, attn_mask,
                        src_key_padding_mask)
        outs = []
        for i in range(0, batch, rows):
            kpm = None if src_key_padding_mask is None else src_key_padding_mask[
                i:i + rows]
            outs.append(
                base(layer, src[:, i:i + rows], pos_emb, chunk_size, attn_mask,
                     kpm))
        return torch.cat(outs, dim=1)

    return forward


def set_attention_budget(model, budget_mb=None):
    """Split each Zipformer2EncoderLayer call along its batch axis so that one
    piece holds at most budget_mb MB of attention scores. The layer treats the
    batch items independently, so the result is the same up to float rounding.
    budget_mb=None restores the upstream call."""
    for m in model.modules():
        if isinstance(m, Zipformer2EncoderLayer):
            if budget_mb is None:
                m.__dict__.pop('forward', None)
            else:
                m.forward = _chunked_forward(m, int(budget_mb * 2**20))
    return model


def _chunked_conv_forward(conv, budget_bytes):
    base = type(conv).forward
    kh, kw = conv.kernel_size
    sh, sw = conv.stride
    dh, dw = conv.dilation

    def forward(x):
        if x.dim() != 4:
            return base(conv, x)
        n, c, h, w = x.shape
        h_out = (h - dh * (kh - 1) - 1) // sh + 1
        w_out = (w - dw * (kw - 1) - 1) // sw + 1
        # CPU im2col column buffer: (n, c * kh * kw, h_out * w_out) values
        per_col = n * c * kh * kw * max(h_out, 1) * x.element_size()
        cols = max(1, budget_bytes // max(per_col, 1))
        if cols >= w_out:
            return base(conv, x)
        outs = []
        for j0 in range(0, w_out, cols):
            j1 = min(w_out, j0 + cols)
            xs = x[..., j0 * sw:(j1 - 1) * sw + dw * (kw - 1) + 1]
            outs.append(
                F.conv2d(xs, conv.weight, conv.bias, conv.stride, 0,
                         conv.dilation, conv.groups))
        return torch.cat(outs, dim=3)

    return forward


def set_conv_budget(model, budget_mb=None):
    """Split the unpadded Conv2d layers (the DenseBlockV2 convolutions and the
    decoder output convolutions) along the frequency axis so that one piece
    needs at most budget_mb MB of CPU im2col buffer. The layers have no padding
    and each output column only reads its own input columns, so the result is
    the same up to float rounding. Useful on CPU without mkldnn (for example
    macOS arm64 wheels); not needed on CUDA. budget_mb=None restores the
    upstream call."""
    for m in model.modules():
        if (isinstance(m, torch.nn.Conv2d) and tuple(m.padding) == (0, 0)
                and m.padding_mode == 'zeros' and m.groups == 1):
            if budget_mb is None:
                m.__dict__.pop('forward', None)
            else:
                m.forward = _chunked_conv_forward(m, int(budget_mb * 2**20))
    return model
