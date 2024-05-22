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
from models.detr import DETR
from models.transformer import Transformer
from models.backbone import Backbone, Joiner
from models.position_encoding import PositionEmbeddingSine


def create_model(weights):
    backbone = build_backbone()
    transformer = Transformer(d_model=256, return_intermediate_dec=True)

    model = DETR(backbone, transformer, num_classes=91, num_queries=100)

    checkpoint = torch.load(weights, map_location='cpu')['model']
    model.load_state_dict(checkpoint)

    return model


def build_backbone():
    position_embedding = PositionEmbeddingSine(num_pos_feats=128, normalize=True)
    backbone = Backbone('resnet50', train_backbone=True, return_interm_layers=False, dilation=False)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
