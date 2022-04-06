# Copyright (c) 2022 Intel Corporation
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
from modnet_onnx import MODNet


def create_modnet(weights):
    model = MODNet(backbone_pretrained=False)

    checkpoint = load(weights, map_location='cpu')
    ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(ckpt)

    return model
