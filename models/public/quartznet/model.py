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
from nemo.collections.asr import JasperEncoder, JasperDecoderForCTC


class QuartzNet(torch.nn.Module):
    def __init__(self, model_config, encoder_weights, decoder_weights):
        super().__init__()
        with open(model_config, 'r') as config:
            model_args = yaml.safe_load(config)
        _ = nemo.core.NeuralModuleFactory(placement=2)

        encoder_params = model_args['init_params']['encoder_params']['init_params']
        self.encoder = JasperEncoder(**encoder_params)
        self.encoder.load_state_dict(torch.load(encoder_weights, map_location='cpu'))

        decoder_params = model_args['init_params']['decoder_params']['init_params']
        num_decoder_input_features = decoder_params['feat_in']
        self.decoder = JasperDecoderForCTC(
            feat_in=num_decoder_input_features,
            num_classes=decoder_params['num_classes'],
            vocabulary=decoder_params['vocabulary'],
        )
        self.decoder.load_state_dict(torch.load(decoder_weights, map_location='cpu'))

        self.encoder._prepare_for_deployment()
        self.decoder._prepare_for_deployment()

        type(self.encoder).__call__ = torch.nn.Module.__call__
        type(self.decoder).__call__ = torch.nn.Module.__call__

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def forward(self, input_signal):
        i_encoded = self.encoder(input_signal)
        i_log_probs = self.decoder(encoder_output=i_encoded)
        return i_log_probs 
