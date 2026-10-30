# -------------------------------------------------------------------------------------------------
# Vendored for IndoML 2026 Track 2: minimal ESPnet speech-enhancement inference code (no espnet import).
# Upstream file : espnet2/enh/separator/bsrnn_separator.py
# Upstream repo : https://github.com/kohei0209/espnet  (branch urgent2025, the URGENT 2025 recipe fork)
#                 commit 6be80495bca76d8406ca11e5971a9954949ac9c1
#                 For this file the fork is byte-identical to https://github.com/espnet/espnet
#                 tag v.202409 (commit 6bae9d295c7113253203087b87d77e28117f0807).
# Upstream blob : git SHA-1 cdca9f346d3a4c7fddcb0f10127d9caf378e88fa
# Copyright 2017 Johns Hopkins University (Shinji Watanabe) and the ESPnet contributors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. A copy of the License is in the file LICENSE in this folder. You can
# also get a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing permissions and limitations.
# [IndoML-T2 change] This file is a modified copy (Apache-2.0 section 4(b)). Changes:
#   - the torch_complex import is removed and the espnet2 imports are changed to local relative imports
#   - type hints Union[torch.Tensor, ComplexTensor] changed to torch.Tensor
#   - for a real-valued input (B, T, F, 2) the output is a torch complex tensor (torch.complex) instead of a torch_complex.ComplexTensor; for a complex input (our case) nothing changes
# -------------------------------------------------------------------------------------------------
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch

from .bsrnn import BSRNN
from .complex_utils import is_complex, new_complex_like
from .abs_separator import AbsSeparator


class BSRNNSeparator(AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        num_spk: int = 1,
        num_channels: int = 16,
        num_layers: int = 6,
        target_fs: int = 48000,
        causal: bool = True,
        norm_type: str = "GN",
        ref_channel: Optional[int] = None,
    ):
        """Band-split RNN (BSRNN) separator.

        Reference:
            [1] J. Yu, H. Chen, Y. Luo, R. Gu, and C. Weng, “High fidelity speech
            enhancement with band-split RNN,” in Proc. ISCA Interspeech, 2023.
            https://isca-speech.org/archive/interspeech_2023/yu23b_interspeech.html
            [2] J. Yu, and Y. Luo, “Efficient monaural speech enhancement with
            universal sample rate band-split RNN,” in Proc. ICASSP, 2023.
            https://ieeexplore.ieee.org/document/10096020

        Args:
            input_dim: (int) maximum number of frequency bins corresponding to
                `target_fs`
            num_spk: (int) number of speakers.
            num_channels: (int) feature dimension in the BandSplit block.
            num_layers: (int) number of processing layers.
            target_fs: (int) max sampling frequency that the model can handle.
            causal (bool): whether or not to apply causal modeling.
                if True, LSTM will be used instead of BLSTM for time modeling
            norm_type (str): type of the normalization layer (cfLN / cLN / BN / GN).
            ref_channel: (int) reference channel. not used for now.
        """
        super().__init__()

        self._num_spk = num_spk
        self.ref_channel = ref_channel

        self.bsrnn = BSRNN(
            input_dim=input_dim,
            num_channel=num_channels,
            num_layer=num_layers,
            target_fs=target_fs,
            causal=causal,
            num_spk=num_spk,
            norm_type=norm_type,
        )

    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]:
        """BSRNN Forward.

        Args:
            input (torch.Tensor or ComplexTensor): STFT spectrum [B, T, (C,) F (,2)]
            ilens (torch.Tensor): input lengths [Batch]
            additional (Dict or None): other data included in model.
                unused in this model.

        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): [(B, T, F), ...]
            ilens (torch.Tensor): (B,)
            others predicted data, e.g. masks: OrderedDict[
                'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'mask_spkn': torch.Tensor(Batch, Frames, Freq),
            ]
        """
        # B, T, (C,) F, 2
        if is_complex(input):
            feature = torch.stack([input.real, input.imag], dim=-1)
        else:
            assert input.size(-1) == 2, input.shape
            feature = input

        opt = {}
        if additional is not None and "fs" in additional:
            opt["fs"] = additional["fs"]
        masked = self.bsrnn(feature, **opt)
        # B, num_spk, T, F
        if not is_complex(input):
            masked = list(torch.complex(masked[..., 0], masked[..., 1]).unbind(1))
        else:
            masked = list(
                new_complex_like(input, (masked[..., 0], masked[..., 1])).unbind(1)
            )

        others = {}
        return masked, ilens, others

    @property
    def num_spk(self):
        return self._num_spk
