# -------------------------------------------------------------------------------------------------
# Vendored for IndoML 2026 Track 2: minimal ESPnet speech-enhancement inference code (no espnet import).
# Upstream file : espnet2/enh/decoder/stft_decoder.py
# Upstream repo : https://github.com/kohei0209/espnet  (branch urgent2025, the URGENT 2025 recipe fork)
#                 commit 6be80495bca76d8406ca11e5971a9954949ac9c1
#                 For this file the fork is byte-identical to https://github.com/espnet/espnet
#                 tag v.202409 (commit 6bae9d295c7113253203087b87d77e28117f0807).
# Upstream blob : git SHA-1 01708898ab1072add6bd3cbcb193823d62242359
# Copyright 2017 Johns Hopkins University (Shinji Watanabe) and the ESPnet contributors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. A copy of the License is in the file LICENSE in this folder. You can
# also get a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing permissions and limitations.
# [IndoML-T2 change] This file is a modified copy (Apache-2.0 section 4(b)). Changes:
#   - imports of torch_complex, packaging and AbsDecoder removed; base class is torch.nn.Module
#   - input must be a torch complex tensor (ComplexTensor is not vendored); the torch version check is removed
#   - the streaming methods (_get_window_func, forward_streaming, streaming_merge) and the __main__ test block are removed
#   - decorator "@torch.cuda.amp.autocast(enabled=False)" changed to "@torch.amp.autocast(\"cuda\", enabled=False)" (same behaviour; the old form is deprecated in new PyTorch; ESPnet v.202610 uses the new form)
# -------------------------------------------------------------------------------------------------
import torch

from .complex_utils import is_torch_complex_tensor
from .stft import Stft



class STFTDecoder(torch.nn.Module):
    """STFT decoder for speech enhancement and separation"""

    def __init__(
        self,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        window="hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        default_fs: int = 16000,
        spec_transform_type: str = None,
        spec_factor: float = 0.15,
        spec_abs_exponent: float = 0.5,
    ):
        super().__init__()
        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
            center=center,
            normalized=normalized,
            onesided=onesided,
        )

        self.win_length = win_length if win_length else n_fft
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
        self.center = center
        self.default_fs = default_fs

        # spec transform related. See equation (1) in paper
        # 'Speech Enhancement and Dereverberation With Diffusion-Based Generative
        # Models'. The default value of 0.15, 0.5 also come from the paper.
        # spec_transform_type: "exponent", "log", or "none"
        self.spec_transform_type = spec_transform_type
        # the output specturm will be scaled with: spec * self.spec_factor
        self.spec_factor = spec_factor
        # the exponent factor used in the "exponent" transform
        self.spec_abs_exponent = spec_abs_exponent

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, input: torch.Tensor, ilens: torch.Tensor, fs: int = None):
        """Forward.

        Args:
            input (ComplexTensor): spectrum [Batch, T, (C,) F]
            ilens (torch.Tensor): input lengths [Batch]
            fs (int): sampling rate in Hz
                If not None, reconfigure iSTFT window and hop lengths for a new
                sampling rate while keeping their duration fixed.
        """
        if not torch.is_complex(input):
            raise TypeError("Only support complex tensors for stft decoder")
        if fs is not None:
            self._reconfig_for_fs(fs)

        input = self.spec_back(input)

        bs = input.size(0)
        if input.dim() == 4:
            multi_channel = True
            # input: (Batch, T, C, F) -> (Batch * C, T, F)
            input = input.transpose(1, 2).reshape(-1, input.size(1), input.size(3))
        else:
            multi_channel = False

        # for supporting half-precision training
        if input.dtype in (torch.float16, torch.bfloat16):
            wav, wav_lens = self.stft.inverse(input.float(), ilens)
            wav = wav.to(dtype=input.dtype)
        elif (
            is_torch_complex_tensor(input)
            and hasattr(torch, "complex32")
            and input.dtype == torch.complex32
        ):
            wav, wav_lens = self.stft.inverse(input.cfloat(), ilens)
            wav = wav.to(dtype=input.dtype)
        else:
            wav, wav_lens = self.stft.inverse(input, ilens)

        if multi_channel:
            # wav: (Batch * C, Nsamples) -> (Batch, Nsamples, C)
            wav = wav.reshape(bs, -1, wav.size(1)).transpose(1, 2)

        self._reset_config()
        return wav, wav_lens

    def _reset_config(self):
        """Reset the configuration of iSTFT window and hop lengths."""
        self._reconfig_for_fs(self.default_fs)

    def _reconfig_for_fs(self, fs):
        """Reconfigure iSTFT window and hop lengths for a new sampling rate

        while keeping their duration fixed.

        Args:
            fs (int): new sampling rate
        """
        self.stft.n_fft = self.n_fft * fs // self.default_fs
        self.stft.win_length = self.win_length * fs // self.default_fs
        self.stft.hop_length = self.hop_length * fs // self.default_fs

    def spec_back(self, spec):
        if self.spec_transform_type == "exponent":
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs() ** (1 / e) * torch.exp(1j * spec.angle())
        elif self.spec_transform_type == "log":
            spec = spec / self.spec_factor
            spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
        elif self.spec_transform_type == "none":
            spec = spec
        return spec
