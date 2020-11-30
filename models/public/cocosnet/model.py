# Copyright (c) 2020 Intel Corporation
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

import argparse
from torch import nn, load, cat

import models.networks as networks


def remove_all_spectral_norm(item):
    if isinstance(item, nn.Module):
        try:
            nn.utils.remove_spectral_norm(item)
        except Exception:
            pass
        for child in item.children():
            remove_all_spectral_norm(child)


class Pix2PixModel(nn.Module):
    def __init__(self, corr_weights, gen_weights):
        super().__init__()
        opt = argparse.Namespace(adaptor_kernel=3,
                                 adaptor_nonlocal=False,
                                 adaptor_res_deeper=False,
                                 adaptor_se=False,
                                 apex=False,
                                 aspect_ratio=1,
                                 CBN_intype='warp_mask',
                                 crop_size=256,
                                 eqlr_sn=False,
                                 gpu_ids=[],
                                 isTrain=False,
                                 init_type='xavier',
                                 init_variance=0.02,
                                 maskmix=True,
                                 mask_noise=False,
                                 match_kernel=3,
                                 netG='spade',
                                 ngf=64,
                                 norm_E='spectralinstance',
                                 norm_G='spectralspadesyncbatch3x3',
                                 noise_for_mask=False,
                                 PONO=True,
                                 PONO_C=True,
                                 semantic_nc=151,
                                 show_corr=False,
                                 show_warpmask=False,
                                 use_attention=True,
                                 use_coordconv=False,
                                 warp_bilinear=False,
                                 warp_cycle_w=0.0,
                                 warp_mask_losstype=True,
                                 warp_patch=False,
                                 warp_stride=4,
                                 weight_domainC=0.0
                                 )
        self.correspondence = networks.define_Corr(opt)
        corr_weights = load(corr_weights)
        self.correspondence.load_state_dict(corr_weights)
        self.generator = networks.define_G(opt)
        gen_weights = load(gen_weights)
        self.generator.load_state_dict(gen_weights)
        remove_all_spectral_norm(self)

    def forward(self, input_semantics, ref_image, ref_semantics):
        coor_out = self.correspondence(ref_image, None, input_semantics, ref_semantics, alpha=1)
        warp_out = cat((coor_out['warp_out'], input_semantics), dim=1)
        return self.generator(input_semantics, warp_out=warp_out)
