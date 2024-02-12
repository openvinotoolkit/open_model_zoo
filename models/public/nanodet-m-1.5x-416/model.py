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
from arch.one_stage_detector import OneStageDetector
from util.config import load_config, cfg


def create_nanodet(cfg_path, weights):
    load_config(cfg, cfg_path)
    model_cfg = cfg.model
    model_cfg.arch.backbone.pop("name")
    model_cfg.arch.fpn.pop("name")
    model_cfg.arch.head.pop("name")
    model = OneStageDetector(model_cfg.arch.backbone, model_cfg.arch.fpn, model_cfg.arch.head)

    checkpoint = load(weights, map_location='cpu')['state_dict']
    ckpt = {k.replace('model.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(ckpt)

    return model
