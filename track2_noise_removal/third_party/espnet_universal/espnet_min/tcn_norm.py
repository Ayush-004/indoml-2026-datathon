# -------------------------------------------------------------------------------------------------
# Vendored for IndoML 2026 Track 2: minimal ESPnet speech-enhancement inference code (no espnet import).
# Upstream file : espnet2/enh/layers/tcn.py
# Upstream repo : https://github.com/kohei0209/espnet  (branch urgent2025, the URGENT 2025 recipe fork)
#                 commit 6be80495bca76d8406ca11e5971a9954949ac9c1
#                 For this file the fork is byte-identical to https://github.com/espnet/espnet
#                 tag v.202409 (commit 6bae9d295c7113253203087b87d77e28117f0807).
# Upstream blob : git SHA-1 b9689ad429cc96bf61bec570d3755bd8fa109302
# Copyright 2017 Johns Hopkins University (Shinji Watanabe) and the ESPnet contributors.
# tcn.py says: "The code is based on: https://github.com/kaituoxu/Conv-TasNet/blob/master/src/conv_tasnet.py
# Licensed under MIT."
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. A copy of the License is in the file LICENSE in this folder. You can
# also get a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing permissions and limitations.
# [IndoML-T2 change] This file is a modified copy (Apache-2.0 section 4(b)). Changes:
#   - only EPS and choose_norm() are kept; all other classes and functions of tcn.py are removed
#   - imports reduced to torch and torch.nn
#   - choose_norm("gLN") and choose_norm("cLN") raise NotImplementedError (their classes are not vendored)
# -------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn

EPS = torch.finfo(torch.get_default_dtype()).eps



def choose_norm(norm_type, channel_size, shape="BDT"):
    """The input of normalization will be (M, C, K), where M is batch size.

    C is channel size and K is sequence length.
    """
    if norm_type in ("gLN", "cLN"):
        # [IndoML-T2 change] GlobalLayerNorm and ChannelwiseLayerNorm are not vendored
        raise NotImplementedError(norm_type)
    elif norm_type == "BN":
        # Given input (M, C, K), nn.BatchNorm1d(C) will accumulate statics
        # along M and K, so this BN usage is right.
        return nn.BatchNorm1d(channel_size)
    elif norm_type == "GN":
        return nn.GroupNorm(1, channel_size, eps=1e-8)
    else:
        raise ValueError("Unsupported normalization type")
