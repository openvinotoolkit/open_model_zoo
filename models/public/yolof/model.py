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
import numpy as np

from models.config import config
from cvpods.layers import ShapeSpec
from cvpods.modeling.backbone import Backbone
from cvpods.modeling.anchor_generator import DefaultAnchorGenerator

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


def build_anchor_generator(cfg, input_shape):
    return DefaultAnchorGenerator(cfg, input_shape)


# def permute_to_N_HWA_K(tensor, K):
#     """
#     Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA), K)
#     """
#     assert tensor.dim() == 4, tensor.shape
#     N, _, H, W = tensor.shape
#     tensor = tensor.view(N, -1, K, H, W)
#     tensor = tensor.permute(0, 3, 4, 1, 2)
#     tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
#     return tensor


class YOLOF(nn.Module):
    """
    Implementation of YOLOF.
    """
    def __init__(self, cfg):
        super().__init__()

        self.device = 'cpu'

        # fmt: off
        self.num_classes = cfg.MODEL.YOLOF.DECODER.NUM_CLASSES
        self.in_features = cfg.MODEL.YOLOF.ENCODER.IN_FEATURES

        # fmt: on
        self.backbone = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.encoder = cfg.build_encoder(
            cfg, backbone_shape
        )
        self.decoder = cfg.build_decoder(cfg)
        self.anchor_generator = cfg.build_anchor_generator(cfg, feature_shapes)

        self.register_buffer(
            "pixel_mean",
            torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        )
        self.register_buffer(
            "pixel_std",
            torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        )
        self.to(self.device)

    def forward(self, images):
        """
        Args:
            inputs[Tensor]: [BxCxHxW]
        Returns:
            outputs[Tensor]: [Bx(H*W*A=6)x(K+4(boxes)+4(anhors))]
        """
        h, w = images.shape[2:]

        features = self.backbone(images)
        features = [features[f] for f in self.in_features]
        box_cls, box_delta = self.decoder(self.encoder(features[0]))

        results = self.inference(box_cls, box_delta)

        return results

    def inference(self, box_cls, box_delta):
        N, _, H, W = box_delta.shape
        result = torch.cat((box_delta.view(N, 6, -1, H, W), torch.ones(N, 6, 1, H, W), box_cls.view(N, 6, -1, H, W)), 2).view(N, -1, H,  W)
        return result


def get_model(weights):
    cfg = config
    print(type(config))
    cfg.build_backbone = build_backbone
    cfg.build_anchor_generator = build_anchor_generator
    cfg.build_encoder = build_encoder
    cfg.build_decoder = build_decoder
    model = YOLOF(cfg)
    checkpoint = torch.load(weights, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    return model
