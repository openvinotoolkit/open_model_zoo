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
from config import cfg
import models


class HpeHRNet(torch.nn.Module):
    def __init__(self, cfg, weights):
        super().__init__()
        self.impl = models.pose_higher_hrnet.PoseHigherResolutionNet(cfg)
        checkpoint = torch.load(weights, map_location='cpu')
        self.impl.load_state_dict(checkpoint)
        self.impl.eval()
        # pooling operation to get nms_heatmaps from heatmaps out of model
        self.pool = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        # ReLU operation to avoid negative values at heatmap
        self.relu = torch.nn.ReLU()

    def forward(self, image):
        outputs = self.impl(image)
        # output[0] - heatmaps_lr_and_embeddings out with size [1, 34, h/4, w/4]
        # output[1] - heatmaps out with size [1, 17, h/2, h/2]
        # resize low-resolution heatmaps and embeddings (outputs[0]) to heatmaps shape (output[1])
        outputs[0] = torch.nn.functional.interpolate(
                outputs[0],
                size=(outputs[-1].size(2), outputs[-1].size(3)),
                mode='bilinear',
                align_corners=False
            )
        # average of heatmaps and apply relu
        outputs[1] = (outputs[0][:, :17, :, :] + outputs[1]) / 2
        outputs[1] = self.relu(outputs[1])
        outputs[0] = outputs[0][:, 17:, :, :]
        # apply nms for heatmaps
        pooled = self.pool(outputs[1])
        mask = torch.eq(pooled, outputs[1]).float()
        mask = mask * 2 - 1
        outputs[1] *= mask
        return outputs


def get_net(file_config, weights):
    cfg.defrost()
    cfg.merge_from_file(file_config)

    model = HpeHRNet(cfg, weights)
    return model
