# Copyright (c) 2022-2024 Intel Corporation
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

from addict import Dict
from cvpods.modeling.backbone import Backbone
from models.cspdarknet import build_darknet_backbone
from yolof_base import build_encoder, build_decoder


def build_backbone(cfg, input_shape=None):
    backbone = build_darknet_backbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone


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
        N, _, H, W = box_delta.shape
        result = torch.cat((box_delta.view(N, 6, -1, H, W), box_cls.view(N, 6, -1, H, W)), 2).view(N, -1, H,  W)

        return result


ConfigDict = Dict()
ConfigDict.MODEL.DARKNET.DEPTH = 53
ConfigDict.MODEL.DARKNET.WITH_CSP = True
ConfigDict.MODEL.DARKNET.NORM = 'BN'
ConfigDict.MODEL.DARKNET.OUT_FEATURES = ['res5']
ConfigDict.MODEL.DARKNET.RES5_DILATION = 2

ConfigDict.MODEL.YOLOF.ENCODER.IN_FEATURES = ['res5']
ConfigDict.MODEL.YOLOF.ENCODER.NUM_CHANNELS = 512
ConfigDict.MODEL.YOLOF.ENCODER.BLOCK_MID_CHANNELS = 128
ConfigDict.MODEL.YOLOF.ENCODER.NUM_RESIDUAL_BLOCKS = 8
ConfigDict.MODEL.YOLOF.ENCODER.BLOCK_DILATIONS = [1, 2, 3, 4, 5, 6, 7, 8]
ConfigDict.MODEL.YOLOF.ENCODER.NORM = 'BN'
ConfigDict.MODEL.YOLOF.ENCODER.ACTIVATION = 'LeakyReLU'

ConfigDict.MODEL.YOLOF.DECODER.IN_CHANNELS = 512
ConfigDict.MODEL.YOLOF.DECODER.NUM_CLASSES = 80
ConfigDict.MODEL.YOLOF.DECODER.NUM_ANCHORS = 6
ConfigDict.MODEL.YOLOF.DECODER.CLS_NUM_CONVS = 2
ConfigDict.MODEL.YOLOF.DECODER.REG_NUM_CONVS = 4
ConfigDict.MODEL.YOLOF.DECODER.NORM = 'BN'
ConfigDict.MODEL.YOLOF.DECODER.ACTIVATION = 'LeakyReLU'
ConfigDict.MODEL.YOLOF.DECODER.PRIOR_PROB = 0.01


def get_model(weights):
    cfg = ConfigDict
    cfg.build_backbone = build_backbone
    cfg.build_encoder = build_encoder
    cfg.build_decoder = build_decoder

    model = YOLOF(cfg)
    checkpoint = torch.load(weights, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    return model
