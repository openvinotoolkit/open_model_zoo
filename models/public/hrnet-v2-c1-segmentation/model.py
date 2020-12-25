# Copyright (c) 2020 Intel Corporation
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

import torch.nn

from mit_semseg.models import SegmentationModule, ModelBuilder

class HrnetV2C1(torch.nn.Module):
    def __init__(self, encoder_weights, decoder_weights):
        super().__init__()
        self.impl = SegmentationModule(
            net_enc=ModelBuilder.build_encoder(
                arch="hrnetv2", fc_dim=720, weights=encoder_weights),
            net_dec=ModelBuilder.build_decoder(
                arch="c1", fc_dim=720, num_class=150, weights=decoder_weights, use_softmax=True),
            crit=None,
        )

    def forward(self, image):
        return self.impl({'img_data': image}, segSize=320)
