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

from torch import nn, load, cat
from options.test_options import TestOptions
import models.networks as networks


def remove_all_spectral_norm(item):
    if isinstance(item, nn.Module):
        try:
            nn.utils.remove_spectral_norm(item)
        except Exception:
            pass
        for child in item.children():
            remove_all_spectral_norm(child)
    if isinstance(item, nn.ModuleList):
        for module in item:
            remove_all_spectral_norm(module)
    if isinstance(item, nn.Sequential):
        modules = item.children()
        for module in modules:
            remove_all_spectral_norm(module)


class Pix2PixModel(nn.Module):
    def __init__(self, corr_weights, gen_weights):
        super().__init__()
        opt = TestOptions().parse()
        opt.name = "ade20k"
        opt.use_attention = True
        opt.maskmix = True
        opt.PONO = True
        opt.PONO_C = True
        opt.batchSize = 1
        opt.warp_mask_losstype = 'direct'
        opt.semantic_nc = 151
        self.correspondence = networks.define_Corr(opt)
        corr_weights = load(corr_weights)
        self.correspondence.load_state_dict(corr_weights)
        self.generator = networks.define_G(opt)
        gen_weights = load(gen_weights)
        self.generator.load_state_dict(gen_weights)
        remove_all_spectral_norm(self)
        print("finish")

    def forward(self, input_semantics, ref_image, ref_semantics):
        coor_out = self.correspondence(ref_image, None, input_semantics, ref_semantics, alpha=1)
        warp_out = cat((coor_out['warp_out'], input_semantics), dim=1)
        return self.generator(input_semantics, warp_out=warp_out)
