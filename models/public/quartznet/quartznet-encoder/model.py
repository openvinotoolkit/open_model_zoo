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
import yaml

import nemo
from nemo.collections.asr import jasper


def quartznet_encoder(model_config, weights):
    _ = nemo.core.NeuralModuleFactory(placement=2)
    with open(model_config, 'r') as config:
        model_args = yaml.safe_load(config)

    encoder_params = model_args['init_params']['encoder_params']['init_params']

    jasper_encoder = jasper.JasperEncoder(**encoder_params)
    jasper_encoder.restore_from(weights)
    _inexample, _out_example = jasper_encoder._prepare_for_deployment()
    type(jasper_encoder).__call__ = torch.nn.Module.__call__

    return jasper_encoder
