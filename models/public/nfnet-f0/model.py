# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from timm.models.nfnet import NfCfg, NormFreeNet


def create_normfreenet():
    channels = (256, 512, 1536, 1536)
    model_cfg = NfCfg(depths=(1, 2, 6, 3), channels=channels, stem_type='deep_quad', stem_chs=128, group_size=128,
                      bottle_ratio=0.5, extra_conv=True, gamma_in_act=True, same_padding=True, skipinit=True,
                      num_features=channels[-1] * 2, act_layer='gelu', attn_layer='se',
                      attn_kwargs={'reduction_ratio': 0.5, 'divisor': 8})

    model = NormFreeNet(cfg=model_cfg)

    return model
