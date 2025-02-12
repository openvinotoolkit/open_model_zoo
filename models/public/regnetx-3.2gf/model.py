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
import pycls.core.config
import pycls.models.model_zoo
from pycls.core.checkpoint import unwrap_model

def regnet(config_path, weights_path):
    pycls.core.config.cfg.merge_from_file(config_path)
    model = pycls.models.model_zoo.RegNet()
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    test_err = checkpoint.get("test_err", 100)
    ema_err = checkpoint.get("ema_err", 100)
    ema_state = "ema_state" if "ema_state" in checkpoint else "model_state"
    best_state = "model_state" if test_err <= ema_err else ema_state
    unwrap_model(model).load_state_dict(checkpoint[best_state])
    return model
