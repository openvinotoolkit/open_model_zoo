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

import torch
from torch import nn

from cvpods.modeling.backbone import Backbone
from models.cspdarknet import build_darknet_backbone
from yolof_base import build_encoder, build_decoder


def build_backbone(cfg, input_shape=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.
    Returns:
        an instance of :class:`Backbone`
    """

    backbone = build_darknet_backbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone

def permute_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


class YOLOF(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = 'cpu'

        self.num_classes = cfg.MODEL.YOLOF.DECODER.NUM_CLASSES
        self.in_features = cfg.MODEL.YOLOF.ENCODER.IN_FEATURES

        self.backbone = cfg.build_backbone(cfg)

        backbone_shape = self.backbone.output_shape()
        self.encoder = cfg.build_encoder(
            cfg, backbone_shape
        )
        self.decoder = cfg.build_decoder(cfg)
        self.to(self.device)

    def forward(self, images):
        """
        Args:
            inputs[Tensor]: [BxCxHxW]
        Returns:
            outputs[Tensor]: [Bx(H*W*A=6)x(K(classes)+4(boxes))]
        """
        h, w = images.shape[2:]

        features = self.backbone(images)
        features = [features[f] for f in self.in_features]
        box_cls, box_delta = self.decoder(self.encoder(features[0]))

        results = self.inference(box_cls, box_delta)

        return results

    def inference(self, box_cls, box_delta):
        box_cls = permute_to_N_HWA_K(box_cls, self.num_classes)
        box_delta = permute_to_N_HWA_K(box_delta, 4)

        N, A, C = box_delta.shape
        result = torch.cat((box_delta, box_cls), 2)

        return result

class ConfigDict(dict):
    def __init__(self, d={}):
        if d == {}:
            d = {
                'MODEL': {
                    'DARKNET': {
                        'DEPTH': 53,
                        'WITH_CSP': True,
                        'NORM': "BN",
                        'OUT_FEATURES': ["res5"],
                        'RES5_DILATION': 2
                    },
                    'YOLOF': {
                        'ENCODER': {
                            'IN_FEATURES': ["res5"],
                            'NUM_CHANNELS': 512,
                            'BLOCK_MID_CHANNELS': 128,
                            'NUM_RESIDUAL_BLOCKS': 8,
                            'BLOCK_DILATIONS': [1, 2, 3, 4, 5, 6, 7, 8],
                            'NORM': "BN",
                            'ACTIVATION': "LeakyReLU"
                        },
                        'DECODER': {
                            'IN_CHANNELS': 512,
                            'NUM_CLASSES': 80,
                            'NUM_ANCHORS': 6,
                            'CLS_NUM_CONVS': 2,
                            'REG_NUM_CONVS': 4,
                            'NORM': "BN",
                            'ACTIVATION': "LeakyReLU",
                            'PRIOR_PROB': 0.01
                        }
                    }
                }
            }
        for k, v in d.items():
            setattr(self, k, v)

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super().__setattr__(name, value)
        super().__setitem__(name, value)

def get_model(weights):
    cfg = ConfigDict()
    cfg.build_backbone = build_backbone
    cfg.build_encoder = build_encoder
    cfg.build_decoder = build_decoder

    model = YOLOF(cfg)
    checkpoint = torch.load(weights, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    return model
