# -------------------------------------------------------------------------------------------------
# Vendored for IndoML 2026 Track 2: minimal ESPnet speech-enhancement inference code (no espnet import).
# Upstream file : espnet2/enh/encoder/stft_encoder.py
# Upstream repo : https://github.com/kohei0209/espnet  (branch urgent2025, the URGENT 2025 recipe fork)
#                 commit 6be80495bca76d8406ca11e5971a9954949ac9c1
#                 For this file the fork is byte-identical to https://github.com/espnet/espnet
#                 tag v.202409 (commit 6bae9d295c7113253203087b87d77e28117f0807).
# Upstream blob : git SHA-1 e71f67a7a14101c1672a0431ad5db61c24be2c2b
# Copyright 2017 Johns Hopkins University (Shinji Watanabe) and the ESPnet contributors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. A copy of the License is in the file LICENSE in this folder. You can
# also get a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing permissions and limitations.
# [IndoML-T2 change] This file is a modified copy (Apache-2.0 section 4(b)). Changes:
#   - imports of packaging, torch_complex and AbsEncoder removed; base class is torch.nn.Module
#   - the torch version check is removed; use_builtin_complex=False raises NotImplementedError
#   - the streaming methods (_apply_window_func, forward_streaming, streaming_frame) are removed
#   - decorator "@torch.cuda.amp.autocast(enabled=False)" changed to "@torch.amp.autocast(\"cuda\", enabled=False)" (same behaviour; the old form is deprecated in new PyTorch; ESPnet v.202610 uses the new form)
# -------------------------------------------------------------------------------------------------
import torch

from .stft import Stft



class STFTEncoder(torch.nn.Module):
    """STFT encoder for speech enhancement and separation"""

    def __init__(
        self,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        window="hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        use_builtin_complex: bool = True,
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

        self._output_dim = n_fft // 2 + 1 if onesided else n_fft
        self.use_builtin_complex = use_builtin_complex
        self.win_length = win_length if win_length else n_fft
        self.hop_length = hop_length
        self.window = window
        self.n_fft = n_fft
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

    def spec_transform_func(self, spec):
        if self.spec_transform_type == "exponent":
            if self.spec_abs_exponent != 1:
                # only do this calculation if spec_exponent != 1,
                # otherwise it's quite a bit of wasted computation
                # and introduced numerical error
                e = self.spec_abs_exponent
                spec = spec.abs() ** e * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.spec_transform_type == "log":
            spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.spec_transform_type == "none":
            spec = spec
        return spec

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, input: torch.Tensor, ilens: torch.Tensor, fs: int = None):
        """Forward.

        Args:
            input (torch.Tensor): mixed speech [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]
            fs (int): sampling rate in Hz
                If not None, reconfigure STFT window and hop lengths for a new
                sampling rate while keeping their duration fixed.
        Returns:
            spectrum (ComplexTensor): [Batch, T, (C,) F]
            flens (torch.Tensor): [Batch]
        """
        if fs is not None:
            self._reconfig_for_fs(fs)
        # for supporting half-precision training
        if input.dtype in (torch.float16, torch.bfloat16):
            spectrum, flens = self.stft(input.float(), ilens)
            spectrum = spectrum.to(dtype=input.dtype)
        else:
            spectrum, flens = self.stft(input, ilens)
        if self.use_builtin_complex:
            spectrum = torch.complex(spectrum[..., 0], spectrum[..., 1])
        else:
            # [IndoML-T2 change] torch_complex.ComplexTensor is not vendored
            raise NotImplementedError("use_builtin_complex=False needs torch_complex")

        self._reset_config()

        spectrum = self.spec_transform_func(spectrum)

        return spectrum, flens

    def _reset_config(self):
        """Reset the configuration of STFT window and hop lengths."""
        self._reconfig_for_fs(self.default_fs)

    def _reconfig_for_fs(self, fs):
        """Reconfigure STFT window and hop lengths for a new sampling rate
        while keeping their duration fixed.

        Args:
            fs (int): new sampling rate
        """  # noqa: H405
        self.stft.n_fft = self.n_fft * fs // self.default_fs
        self.stft.win_length = self.win_length * fs // self.default_fs
        self.stft.hop_length = self.hop_length * fs // self.default_fs
