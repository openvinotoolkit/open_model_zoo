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

from torch import load
import torch.nn as nn
from models.yolox import YOLOX
from models.yolo_pafpn import YOLOPAFPN
from models.yolo_head import YOLOXHead
from models.network_blocks import SiLU


def create_model(weights):
    in_channels = [256, 512, 1024]
    num_classes = 80
    depth = 0.33
    width = 0.375
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels)
    head = YOLOXHead(num_classes, width, in_channels=in_channels)
    model = YOLOX(backbone, head)

    model.apply(init_yolo)

    checkpoint = load(weights, map_location='cpu')['model']
    model.load_state_dict(checkpoint)
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = False

    return model


def init_yolo(M):
    for m in M.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03


def replace_module(module, replaced_module_type, new_module_type):
    model = module
    if isinstance(module, replaced_module_type):
        model = new_module_type()
    else:
        for name, child in module.named_children():
            new_child = replace_module(child, replaced_module_type, new_module_type)
            if new_child is not child:
                model.add_module(name, new_child)
    return model
