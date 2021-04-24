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
from vedastr.utils import Config, build_from_cfg
from vedastr.models.registry import MODELS


def get_model(file_config, weights):
    cfg = Config.fromfile(file_config)
    deploy_cfg = cfg['deploy']

    model = build_from_cfg(deploy_cfg['model'], MODELS)

    checkpoint = torch.load(weights, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    return model
