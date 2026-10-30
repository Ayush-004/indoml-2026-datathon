# -------------------------------------------------------------------------------------------------
# Vendored for IndoML 2026 Track 2: minimal ESPnet speech-enhancement inference code (no espnet import).
# Upstream file : espnet2/layers/stft.py
# Upstream repo : https://github.com/kohei0209/espnet  (branch urgent2025, the URGENT 2025 recipe fork)
#                 commit 6be80495bca76d8406ca11e5971a9954949ac9c1
#                 For this file the fork is byte-identical to https://github.com/espnet/espnet
#                 tag v.202409 (commit 6bae9d295c7113253203087b87d77e28117f0807).
# Upstream blob : git SHA-1 de466714c385392aba5eb63a464cc785e1da4572
# Copyright 2017 Johns Hopkins University (Shinji Watanabe) and the ESPnet contributors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. A copy of the License is in the file LICENSE in this folder. You can
# also get a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing permissions and limitations.
# [IndoML-T2 change] This file is a modified copy (Apache-2.0 section 4(b)). Changes:
#   - imports of librosa, numpy, packaging, torch_complex, typeguard, InversibleInterface and espnet make_pad_mask removed; to_complex comes from the local complex_utils
#   - class Stft has no InversibleInterface base and no @typechecked decorator
#   - the librosa fallback for old PyTorch without MKL is removed; torch.stft is always used (the same as ESPnet v.202610)
#   - type hint Union[torch.Tensor, ComplexTensor] of inverse() changed to torch.Tensor
#   - a small local make_pad_mask() is added at the end of the file
# -------------------------------------------------------------------------------------------------
from typing import Optional, Tuple, Union

import torch

from .complex_utils import to_complex



class Stft(torch.nn.Module):
    def __init__(
        self,
        n_fft: int = 512,
        win_length: Optional[int] = None,
        hop_length: int = 128,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        if win_length is None:
            self.win_length = n_fft
        else:
            self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        if window is not None and not hasattr(torch, f"{window}_window"):
            raise ValueError(f"{window} window is not implemented")
        self.window = window

    def extra_repr(self):
        return (
            f"n_fft={self.n_fft}, "
            f"win_length={self.win_length}, "
            f"hop_length={self.hop_length}, "
            f"center={self.center}, "
            f"normalized={self.normalized}, "
            f"onesided={self.onesided}"
        )

    def forward(
        self, input: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """STFT forward function.

        Args:
            input: (Batch, Nsamples) or (Batch, Nsample, Channels)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Freq, 2) or (Batch, Frames, Channels, Freq, 2)

        """
        bs = input.size(0)
        if input.dim() == 3:
            multi_channel = True
            # input: (Batch, Nsample, Channels) -> (Batch * Channels, Nsample)
            input = input.transpose(1, 2).reshape(-1, input.size(1))
        else:
            multi_channel = False

        # NOTE(kamo):
        #   The default behaviour of torch.stft is compatible with librosa.stft
        #   about padding and scaling.
        #   Note that it's different from scipy.signal.stft

        # output: (Batch, Freq, Frames, 2=real_imag)
        # or (Batch, Channel, Freq, Frames, 2=real_imag)
        if self.window is not None:
            window_func = getattr(torch, f"{self.window}_window")
            window = window_func(
                self.win_length, dtype=input.dtype, device=input.device
            )
        else:
            window = None

        stft_kwargs = dict(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            center=self.center,
            window=window,
            normalized=self.normalized,
            onesided=self.onesided,
        )
        stft_kwargs["return_complex"] = True
        # NOTE(Jinchuan) CuFFT is not compatible with bfloat16
        output = torch.stft(input.float(), **stft_kwargs)
        output = torch.view_as_real(output).type(input.dtype)

        # output: (Batch, Freq, Frames, 2=real_imag)
        # -> (Batch, Frames, Freq, 2=real_imag)
        output = output.transpose(1, 2)
        if multi_channel:
            # output: (Batch * Channel, Frames, Freq, 2=real_imag)
            # -> (Batch, Frame, Channel, Freq, 2=real_imag)
            output = output.view(bs, -1, output.size(1), output.size(2), 2).transpose(
                1, 2
            )

        if ilens is not None:
            if self.center:
                pad = self.n_fft // 2
                ilens = ilens + 2 * pad

            olens = (
                torch.div(ilens - self.n_fft, self.hop_length, rounding_mode="trunc")
                + 1
            )
            output.masked_fill_(make_pad_mask(olens, output, 1), 0.0)
        else:
            olens = None

        return output, olens

    def inverse(
        self, input: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Inverse STFT.

        Args:
            input: Tensor(batch, T, F, 2) or ComplexTensor(batch, T, F)
            ilens: (batch,)
        Returns:
            wavs: (batch, samples)
            ilens: (batch,)
        """
        input = to_complex(input)

        if self.window is not None:
            window_func = getattr(torch, f"{self.window}_window")
            datatype = input.real.dtype
            window = window_func(self.win_length, dtype=datatype, device=input.device)
        else:
            window = None

        input = input.transpose(1, 2)

        wavs = torch.functional.istft(
            input,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            normalized=self.normalized,
            onesided=self.onesided,
            length=ilens.max() if ilens is not None else ilens,
            return_complex=False,
        )

        return wavs, ilens


def make_pad_mask(lengths, xs, length_dim):
    """[IndoML-T2 change] Small replacement of espnet.nets.pytorch_backend.nets_utils.make_pad_mask.

    Only the call of Stft.forward is supported: 1-D LongTensor lengths (B,), reference tensor xs,
    length_dim > 0. The result is True at the padded positions and has the shape of xs.
    """
    if length_dim < 0:
        length_dim = xs.dim() + length_dim
    if length_dim <= 0:
        raise ValueError("length_dim must point to a non-batch axis: {}".format(length_dim))
    maxlen = xs.size(length_dim)
    seq = torch.arange(maxlen, dtype=torch.int64, device=xs.device)
    mask = seq.unsqueeze(0) >= lengths.to(device=xs.device, dtype=torch.int64).unsqueeze(-1)
    shape = [1] * xs.dim()
    shape[0] = xs.size(0)
    shape[length_dim] = maxlen
    return mask.view(shape).expand_as(xs)
