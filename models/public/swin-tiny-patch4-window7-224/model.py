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
from torch.onnx.symbolic_helper import parse_args, _slice_helper
from sys import maxsize as maxsize
from timm.models.swin_transformer import swin_tiny_patch4_window7_224


def create_model(weights):
    try:
        torch.onnx.symbolic_registry.register_op('roll', roll, '', version=11)
    except AttributeError:
        torch.onnx.register_custom_op_symbolic('::roll', roll, 11)
    model = swin_tiny_patch4_window7_224()

    checkpoint = torch.load(weights, map_location='cpu')['model']
    model.load_state_dict(checkpoint)

    return model


@parse_args('v', 'is', 'is')
def roll(g, input, shifts, dims):
    assert len(shifts) == len(dims)
    result = input
    for i in range(len(shifts)):
        shapes = []
        shape = _slice_helper(g, result, axes=[dims[i]], starts=[-shifts[i]], ends=[maxsize])
        shapes.append(shape)
        shape = _slice_helper(g, result, axes=[dims[i]], starts=[0], ends=[-shifts[i]])
        shapes.append(shape)
        result = g.op("Concat", *shapes, axis_i=dims[i])
    return result
