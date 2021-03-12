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

import mmcv
import torch

from functools import partial
from mmcv.onnx import register_extra_symbolics
from mmseg.models import build_segmentor


def _convert_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


def pspnet(weights_path, config_path):
    cfg = mmcv.Config.fromfile(config_path)
    cfg.model.pretrained = None
    segmentor = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    # convert SyncBN to BN
    segmentor = _convert_batchnorm(segmentor)
    img_metas = [[{
        'img_shape': (512, 512, 3),
        'ori_shape': (512, 512, 3),
        'pad_shape': (512, 512, 3),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False,
    }]]

    segmentor.forward = partial(segmentor.forward, img_metas=img_metas, return_loss=False)
    opset_version = 11
    register_extra_symbolics(opset_version)
    weights = torch.load(weights_path, map_location='cpu')
    segmentor.load_state_dict(weights['state_dict'])
    return segmentor