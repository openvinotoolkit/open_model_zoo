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


def quartznet_decoder(model_config, weights):
    _ = nemo.core.NeuralModuleFactory(placement=2)
    with open(model_config, 'r') as config:
        model_args = yaml.safe_load(config)

    decoder_params = model_args['init_params']['decoder_params']['init_params']
    num_decoder_input_features = decoder_params['feat_in']

    jasper_decoder = jasper.JasperDecoderForCTC(
        feat_in=num_decoder_input_features,
        num_classes=decoder_params['num_classes'],
        vocabulary=decoder_params['vocabulary'],
    )
    jasper_decoder.restore_from(weights)
    _inexample, _out_example = jasper_decoder._prepare_for_deployment()
    type(jasper_decoder).__call__ = torch.nn.Module.__call__

    return jasper_decoder
