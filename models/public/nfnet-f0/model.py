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


def set_default_kwargs(names, default_cfg):
    kwargs = {}
    for n in names:
        if n in ('img_size', 'in_chans'):
            input_size = default_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 3
                value = input_size[-2:] if n == 'img_size' else input_size[0]
                kwargs.setdefault(n, value)
        else:
            default_val = default_cfg.get(n, None)
            if default_val is not None:
                kwargs.setdefault(n, default_cfg[n])
    return kwargs


DeepMindNFNetF0Config = {
    'num_classes': 1000,
    'input_size': (3, 192, 192),
    'pool_size': (6, 6),
    'crop_pct': .9,
    'interpolation': 'bicubic',
    'mean': (0.485, 0.456, 0.406),
    'std': (0.229, 0.224, 0.225),
    'first_conv': 'stem.conv1',
    'classifier': 'head.fc',
    'test_input_size': (3, 256, 256),
    'architecture': 'dm_nfnet_f0'
}


def create_normfreenet():
    attn_kwargs = dict(reduction_ratio=0.5, divisor=8)
    channels = (256, 512, 1536, 1536)
    model_cfg = NfCfg(depths=(1, 2, 6, 3), channels=channels, stem_type='deep_quad', stem_chs=128, group_size=128,
                      bottle_ratio=0.5, extra_conv=True, gamma_in_act=True, same_padding=True, skipinit=True,
                      num_features=int(channels[-1] * 2.0), act_layer='gelu', attn_layer='se', attn_kwargs=attn_kwargs)

    default_cfg = DeepMindNFNetF0Config
    kwargs = set_default_kwargs(names=('num_classes', 'global_pool', 'in_chans'), default_cfg=default_cfg)

    model = NormFreeNet(cfg=model_cfg, **kwargs)
    model.default_cfg = default_cfg

    return model
