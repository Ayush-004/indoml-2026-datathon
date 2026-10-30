# -------------------------------------------------------------------------------------------------
# Vendored for IndoML 2026 Track 2: minimal ESPnet speech-enhancement inference code (no espnet import).
# Derived from  : espnet2/enh/layers/complex_utils.py
# Upstream repo : https://github.com/kohei0209/espnet  (branch urgent2025)
#                 commit 6be80495bca76d8406ca11e5971a9954949ac9c1
#                 (byte-identical to espnet/espnet tag v.202409, commit 6bae9d295c71)
# Upstream blob : git SHA-1 f73e8b2e6569d4c32c4fb5e48f0bfb9b1c670584
# Copyright 2017 Johns Hopkins University (Shinji Watanabe) and the ESPnet contributors.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. A copy of the License is in the file LICENSE in this folder. You can
# also get a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing permissions and limitations.
# [IndoML-T2 change] This file is a modified copy (Apache-2.0 section 4(b)). Changes:
#   - only new_complex_like, is_torch_complex_tensor, is_complex and to_complex are kept
#   - the torch_complex (ComplexTensor) branches are removed: only torch builtin complex tensors are
#     supported (the configs of our two models use use_builtin_complex: true)
# -------------------------------------------------------------------------------------------------
"""Complex tensor helpers (torch builtin complex tensors only)."""

from typing import Tuple

import torch


def new_complex_like(
    ref: torch.Tensor,
    real_imag: Tuple[torch.Tensor, torch.Tensor],
):
    if is_torch_complex_tensor(ref):
        return torch.complex(*real_imag)
    else:
        raise ValueError(
            "Please update your PyTorch version to 1.9+ for complex support."
        )


def is_torch_complex_tensor(c):
    return torch.is_complex(c)


def is_complex(c):
    return is_torch_complex_tensor(c)


def to_complex(c):
    # Convert to torch native complex
    if torch.is_complex(c):
        return c
    else:
        return torch.view_as_complex(c)
