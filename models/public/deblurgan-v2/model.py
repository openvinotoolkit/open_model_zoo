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

from models.networks import get_generator


def remove_all_batch_norm(item):
    if isinstance(item, nn.Module):
        for index, child in enumerate(item.children()):
            if isinstance(child, nn.BatchNorm2d):
                item[index] = batch_noramlization(child.bias, child.weight, child.eps)
            else:
                remove_all_batch_norm(child)


class batch_noramlization(nn.Module):
    def __init__(self, bias, weight, eps):
        super().__init__()
        self.bias = bias
        self.weight = weight
        self.eps = eps

    def __call__(self, x):
        mean_cur = torch.mean(x, dim=(0, 2, 3))
        var_cur = torch.std(x, dim=(0, 2, 3), unbiased=False)**2
        invstd = 1 / torch.sqrt(var_cur[:, None, None] + self.eps)
        out = self.weight[:, None, None] * (x - mean_cur[:, None, None]) * invstd + self.bias[:, None, None]
        return out


class DeblurV2(nn.Module):
    def __init__(self, weights, model_name):
        super().__init__()

        parameters = {'g_name': model_name, 'norm_layer': 'instance'}
        self.impl = get_generator(parameters)
        checkpoint = torch.load(weights, map_location='cpu')['model']
        self.impl.load_state_dict(checkpoint)
        self.impl.train(True)
        remove_all_batch_norm(self.impl)

    def forward(self, image):
        out = self.impl(image)
        # convert out to [0, 1] range and change channel order RGB->BGR
        out = (out + 1) / 2
        permute = [2, 1, 0]
        return out[:, permute, :, :]
