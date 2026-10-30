# -------------------------------------------------------------------------------------------------
# Vendored for IndoML 2026 Track 2: minimal ESPnet speech-enhancement inference code (no espnet import).
# Derived from  : espnet2/enh/espnet_model.py   (ESPnetEnhancementModel: constructor attributes)
#                 git SHA-1 1ebc4fee0f50ebe2899f36b460d65c9661a1fc97
#                 espnet2/tasks/enh.py          (EnhancementTask.build_model)
#                 git SHA-1 a7e73306b16824d9797c51275bc3e264a37863cc
#                 espnet2/bin/enh_inference.py  (SeparateSpeech.__init__ and SeparateSpeech.__call__)
#                 git SHA-1 f9b6d54cdae805750166f84223a57f80f98d959b
# Upstream repo : https://github.com/kohei0209/espnet  (branch urgent2025)
#                 commit 6be80495bca76d8406ca11e5971a9954949ac9c1
#                 (byte-identical to espnet/espnet tag v.202409, commit 6bae9d295c71)
# Copyright 2017 Johns Hopkins University (Shinji Watanabe) and the ESPnet contributors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. A copy of the License is in the file LICENSE in this folder. You can
# also get a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing permissions and limitations.
# [IndoML-T2 change] This file is a new, reduced implementation of the parts above (Apache-2.0
# section 4(b)). Changes against upstream:
#   - EnhModel keeps only the attributes that inference uses (no loss wrappers, no forward/loss code)
#   - build_model supports only encoder "stft", decoder "stft" and separators "tfgridnetv3", "bsrnn"
#   - the weights are loaded with strict=True (upstream uses strict=False)
#   - separate_speech does the steps of SeparateSpeech.__call__ for a single-channel input without a
#     category; the segment-wise path does not solve the output permutation (it is not necessary for
#     one output source; num_spk > 1 raises NotImplementedError); always_forward_in_48k is not supported
#   - new helpers: enhance_tensor (the same whole-clip math without torch.no_grad and numpy
#     conversion), set_bsrnn_groupnorm_eps and set_lstm_batch_chunk (see their docstrings)
# -------------------------------------------------------------------------------------------------
"""Minimal ESPnet-SE model wrapper, model builder and SeparateSpeech-equivalent inference."""

from typing import List, Optional, Union

import numpy as np
import torch

from .bsrnn_separator import BSRNNSeparator
from .stft_decoder import STFTDecoder
from .stft_encoder import STFTEncoder
from .tfgridnetv3_separator import TFGridNetV3

ENCODER_CHOICES = {"stft": STFTEncoder}
SEPARATOR_CHOICES = {"tfgridnetv3": TFGridNetV3, "bsrnn": BSRNNSeparator}
DECODER_CHOICES = {"stft": STFTDecoder}
# SeparateSpeech sets sfi_processing = True for BSRNNSeparator, USESSeparator and TFGridNetV3
# (sampling-frequency-independent models). USESSeparator is not vendored.
SFI_SEPARATORS = (BSRNNSeparator, TFGridNetV3)


class EnhModel(torch.nn.Module):
    """Inference part of espnet2.enh.espnet_model.ESPnetEnhancementModel.

    The state_dict keys are the same as in ESPnet ("encoder.*", "separator.*", "decoder.*").
    """

    def __init__(
        self,
        encoder,
        separator,
        decoder,
        mask_module=None,
        loss_wrappers=None,
        stft_consistency: bool = False,
        loss_type: str = "mask_mse",
        mask_type: Optional[str] = None,
        flexible_numspk: bool = False,
        extract_feats_in_collect_stats: bool = False,
        normalize_variance: bool = False,
        normalize_variance_per_ch: bool = False,
        categories: list = (),
        category_weights: list = (),
        always_forward_in_48k: bool = False,
    ):
        super().__init__()
        if mask_module is not None:
            raise NotImplementedError("mask_module is not vendored")
        self.encoder = encoder
        self.separator = separator
        self.decoder = decoder
        self.mask_module = None
        self.num_spk = separator.num_spk

        # for multi-channel signal
        self.ref_channel = getattr(self.separator, "ref_channel", None)
        if self.ref_channel is None:
            self.ref_channel = 0

        self.normalize_variance = normalize_variance
        self.normalize_variance_per_ch = normalize_variance_per_ch
        if normalize_variance and normalize_variance_per_ch:
            raise ValueError(
                "normalize_variance and normalize_variance_per_ch cannot be True "
                "at the same time."
            )

        # list all possible categories of the batch (order matters!)
        self.categories = {}
        if categories:
            count = 0
            for c in categories:
                if c not in self.categories:
                    self.categories[count] = c
                    count += 1

        self.always_forward_in_48k = always_forward_in_48k


def build_model(args: dict) -> EnhModel:
    """Same steps as EnhancementTask.build_model for the supported classes."""
    if args.get("diffusion_model", None) is not None:
        raise NotImplementedError("diffusion models are not vendored")
    encoder = ENCODER_CHOICES[args["encoder"]](**(args.get("encoder_conf") or {}))
    separator = SEPARATOR_CHOICES[args["separator"]](
        encoder.output_dim, **(args.get("separator_conf") or {})
    )
    decoder = DECODER_CHOICES[args["decoder"]](**(args.get("decoder_conf") or {}))
    if args["separator"].endswith("nomask"):
        raise NotImplementedError("mask modules are not vendored")
    return EnhModel(
        encoder=encoder, separator=separator, decoder=decoder, **(args.get("model_conf") or {})
    )


def load_train_config(config_file) -> dict:
    import yaml

    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model_from_file(config_file, model_file=None, device: str = "cpu", dtype: str = "float32"):
    """Build the model from an ESPnet config.yaml and load an ESPnet .pth state_dict (strict).

    Returns (model in eval mode on device, train_args dict).
    """
    args = load_train_config(config_file)
    model = build_model(args)
    if model_file is not None:
        state = torch.load(model_file, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=getattr(torch, dtype)).eval()
    return model, args


def set_bsrnn_groupnorm_eps(model: torch.nn.Module, eps: float) -> int:
    """Set eps of the GroupNorm layers in BSRNN BandSplit and MaskDecoder. Returns their number.

    ESPnet since v.202409 builds these layers with tcn.choose_norm("GN") = GroupNorm(1, C, eps=1e-8).
    The code that trained the published BSRNN weights (https://github.com/Emrys365/se-scaling, on top
    of ESPnet 202304) built them with GroupNorm(1, C), i.e. the PyTorch default eps=1e-5.
    eps is not in the state_dict, so both values load the same weights. The time and frequency
    GroupNorm layers of BSRNN use eps=1e-5 in both versions and are not changed here.
    """
    n = 0
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.GroupNorm) and (".band_split." in name or ".mask_decoder." in name):
            m.eps = float(eps)
            n += 1
    return n


def _chunked_lstm_forward(lstm: torch.nn.LSTM, chunk: int):
    orig = type(lstm).forward

    def forward(input, hx=None):
        if hx is not None or not lstm.batch_first or input.dim() != 3 or input.size(0) <= chunk:
            return orig(lstm, input, hx)
        outs, hs, cs = [], [], []
        for part in input.split(chunk, dim=0):
            o, (h, c) = orig(lstm, part)
            outs.append(o)
            hs.append(h)
            cs.append(c)
        return torch.cat(outs, dim=0), (torch.cat(hs, dim=1), torch.cat(cs, dim=1))

    return forward


def set_lstm_batch_chunk(model: torch.nn.Module, chunk: Optional[int]) -> int:
    """Run each batch_first nn.LSTM of the model on at most `chunk` sequences at a time.

    [IndoML-T2 addition] The sequences of one LSTM batch are independent, so the result is the same
    (only float rounding can change). The CPU LSTM keeps the input projections of all time steps for
    all sequences; this option makes that peak memory smaller. TF-GridNet runs its intra-frame LSTM on
    (frames) sequences and its inter-frame LSTM on (frequency bins) sequences. chunk=None gives the
    normal nn.LSTM forward again. The state_dict does not change. Returns the number of LSTM layers.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, torch.nn.LSTM):
            if chunk is None:
                m.__dict__.pop("forward", None)
            else:
                m.forward = _chunked_lstm_forward(m, int(chunk))
            n += 1
    return n


def _normalize(model, speech_mix):
    """Variance normalisation of SeparateSpeech.__call__. Returns (normalized, mix_std_ or None)."""
    mix_std_ = None
    if getattr(model, "normalize_variance_per_ch", False):
        dim = 1
        mix_std_ = torch.std(speech_mix, dim=dim, keepdim=True)
        speech_mix = speech_mix / mix_std_  # RMS normalization
    elif getattr(model, "normalize_variance", False):
        if speech_mix.ndim > 2:
            dim = (1, 2)
        else:
            dim = 1
        mix_std_ = torch.std(speech_mix, dim=dim, keepdim=True)
        speech_mix = speech_mix / mix_std_  # RMS normalization
    return speech_mix, mix_std_


def _denormalize(model, waves, mix_std_):
    """De-normalisation of SeparateSpeech.__call__."""
    if getattr(model, "normalize_variance_per_ch", False):
        if mix_std_.ndim > 2:
            mix_std_ = mix_std_[:, :, model.ref_channel]
        waves = [w * mix_std_ for w in waves]
    elif getattr(model, "normalize_variance", False):
        if mix_std_.ndim > 2:
            mix_std_ = mix_std_.squeeze(2)
        waves = [w * mix_std_ for w in waves]
    return waves


def _forward_whole(model, speech_mix, lengths, fs_, additional):
    feats, f_lens = model.encoder(speech_mix, lengths, fs=fs_)
    feats, _, _ = model.separator(feats, f_lens, additional)
    return [model.decoder(f, lengths, fs=fs_)[0] for f in feats]


@torch.no_grad()
def separate_speech(
    model: EnhModel,
    speech_mix: Union[torch.Tensor, np.ndarray],
    fs: int = 8000,
    segment_size: Optional[float] = None,
    hop_size: Optional[float] = None,
    normalize_segment_scale: bool = False,
    normalize_output_wav: bool = False,
    device: str = "cpu",
    dtype: str = "float32",
) -> List[np.ndarray]:
    """The steps of espnet2.bin.enh_inference.SeparateSpeech.__call__ (single channel, no category).

    Args:
        speech_mix: (Batch, Nsamples)
        fs: sample rate
    Returns:
        [separated_audio1, ...], each a numpy array (Batch, Nsamples)
    """
    if isinstance(speech_mix, np.ndarray):
        speech_mix = torch.as_tensor(speech_mix)

    assert speech_mix.dim() > 1, speech_mix.size()
    batch_size = speech_mix.size(0)
    speech_mix = speech_mix.to(getattr(torch, dtype))
    # lengths: (B,)
    lengths = speech_mix.new_full(
        [batch_size], dtype=torch.long, fill_value=speech_mix.size(1)
    )

    if isinstance(model.separator, SFI_SEPARATORS):
        fs_ = fs
    else:
        fs_ = None
        if model.always_forward_in_48k:
            raise NotImplementedError("always_forward_in_48k is not vendored")

    # a. To device
    speech_mix = speech_mix.to(device)
    lengths = lengths.to(device)

    # Normalize the signal variance
    speech_mix, mix_std_ = _normalize(model, speech_mix)

    additional = {}

    if segment_size is not None and hop_size is not None and lengths[0] > segment_size * fs:
        if model.num_spk != 1:
            raise NotImplementedError("the permutation solver for num_spk > 1 is not vendored")
        # Segment-wise speech enhancement (same steps as upstream)
        overlap_length = int(np.round(fs * (segment_size - hop_size)))
        num_segments = int(
            np.ceil((speech_mix.size(1) - overlap_length) / (hop_size * fs))
        )
        t = T = int(segment_size * fs)
        pad_shape = speech_mix[:, :T].shape
        enh_waves = []
        for i in range(num_segments):
            st = int(i * hop_size * fs)
            en = st + T
            if en >= lengths[0]:
                # en - st < T (last segment)
                en = lengths[0]
                speech_seg = speech_mix.new_zeros(pad_shape)
                t = en - st
                speech_seg[:, :t] = speech_mix[:, st:en]
            else:
                t = T
                speech_seg = speech_mix[:, st:en]  # B x T [x C]

            lengths_seg = speech_mix.new_full(
                [batch_size], dtype=torch.long, fill_value=T
            )
            # b. Enhancement/Separation Forward
            processed_wav = _forward_whole(model, speech_seg, lengths_seg, fs_, additional)
            if speech_seg.dim() > 2:
                # multi-channel speech
                speech_seg_ = speech_seg[:, model.ref_channel]
            else:
                speech_seg_ = speech_seg

            if normalize_segment_scale:
                # normalize the scale to match the input mixture scale
                mix_energy = torch.sqrt(
                    torch.mean(speech_seg_[:, :t].pow(2), dim=1, keepdim=True)
                )
                enh_energy = torch.sqrt(
                    torch.mean(sum(processed_wav)[:, :t].pow(2), dim=1, keepdim=True)
                )
                processed_wav = [w * (mix_energy / enh_energy) for w in processed_wav]
            # List[torch.Tensor(num_spk, B, T)]
            enh_waves.append(torch.stack(processed_wav, dim=0))

        # c. Stitch the enhanced segments together
        waves = enh_waves[0]
        for i in range(1, num_segments):
            # (one output source: the permutation is the identity)
            if i == num_segments - 1:
                enh_waves[i][:, :, t:] = 0
                enh_waves_res_i = enh_waves[i][:, :, overlap_length:t]
            else:
                enh_waves_res_i = enh_waves[i][:, :, overlap_length:]

            # overlap-and-add (average over the overlapped part)
            waves[:, :, -overlap_length:] = (
                waves[:, :, -overlap_length:] + enh_waves[i][:, :, :overlap_length]
            ) / 2
            # concatenate the residual parts of the later segment
            waves = torch.cat([waves, enh_waves_res_i], dim=2)
        # ensure the stitched length is same as input
        assert waves.size(2) == speech_mix.size(1), (waves.shape, speech_mix.shape)
        waves = torch.unbind(waves, dim=0)
    else:
        # b. Enhancement/Separation Forward
        waves = _forward_whole(model, speech_mix, lengths, fs_, additional)

    # De-normalize the signal variance
    waves = _denormalize(model, waves, mix_std_)

    assert len(waves) == model.num_spk, len(waves) == model.num_spk
    assert len(waves[0]) == batch_size, (len(waves[0]), batch_size)
    if normalize_output_wav:
        waves = [
            (w / abs(w).max(dim=1, keepdim=True)[0] * 0.9).cpu().numpy() for w in waves
        ]  # list[(batch, sample)]
    else:
        waves = [w.cpu().numpy() for w in waves]

    return waves


def enhance_tensor(model: EnhModel, speech_mix: torch.Tensor, fs: int = 16000) -> torch.Tensor:
    """Whole-clip path of separate_speech as a torch function (differentiable, no numpy step).

    speech_mix: (B, N) tensor on the model device. Returns (B, N) for the first output source,
    at the level of the input (no output peak normalisation). All rows must have N valid samples.
    """
    lengths = speech_mix.new_full([speech_mix.size(0)], dtype=torch.long, fill_value=speech_mix.size(1))
    fs_ = fs if isinstance(model.separator, SFI_SEPARATORS) else None
    x, mix_std_ = _normalize(model, speech_mix)
    waves = _forward_whole(model, x, lengths, fs_, {})
    waves = _denormalize(model, waves, mix_std_)
    return waves[0]
