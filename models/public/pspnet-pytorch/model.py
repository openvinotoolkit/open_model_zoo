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

from mmcv.onnx import register_extra_symbolics
from mmseg.models.segmentors import EncoderDecoder


register_extra_symbolics(opset=11)


def _convert_batchnorm(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.SyncBatchNorm):
            new_child  = torch.nn.BatchNorm2d(child.num_features, child.eps,
                                                 child.momentum, child.affine,
                                                 child.track_running_stats)
            if child.affine:
                new_child.weight.data = child.weight.data.clone().detach()
                new_child.bias.data = child.bias.data.clone().detach()
                # keep requires_grad unchanged
                new_child.weight.requires_grad = child.weight.requires_grad
                new_child.bias.requires_grad = child.bias.requires_grad
            new_child.running_mean = child.running_mean
            new_child.running_var = child.running_var
            new_child.num_batches_tracked = child.num_batches_tracked
            setattr(module, name, new_child)
        else:
            _convert_batchnorm(child)


class PSPNet(EncoderDecoder):
    def __init__(self, weights_path, config_path):
        cfg = mmcv.Config.fromfile(config_path)
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        cfg.model.test_cfg = mmcv.Config({'mode': 'whole'})
        del cfg.model.type
        super().__init__(**cfg.model)
        # convert SyncBN to BN
        _convert_batchnorm(self)

        weights = torch.load(weights_path, map_location='cpu')
        self.load_state_dict(weights['state_dict'])

    def forward(self, img):
        img_shape = img.shape[:0:-1]
        img_metas = [[{
            'img_shape': img_shape,
            'ori_shape': img_shape,
            'pad_shape': img_shape,
            'filename': '<demo>.png',
            'scale_factor': 1.0,
            'flip': False,
        }]]
        return super().forward_test([img], img_metas)
