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
import torch.nn as nn
from ruamel.yaml import YAML

from nemo.collections.asr import JasperEncoder, JasperDecoderForCTC
from nemo.core import NeuralModuleFactory, DeviceType

YAML = YAML(typ='safe')

def convert_to_2d(model):
    for name, l in model.named_children():
        layer_type = l.__class__.__name__
        if layer_type == 'Conv1d':
            new_layer = nn.Conv2d(l.in_channels, l.out_channels,
                                  (1, l.kernel_size[0]), (1, l.stride[0]),
                                  (0, l.padding[0]), (1, l.dilation[0]),
                                  l.groups, False if l.bias is None else True, l.padding_mode)
            params = l.state_dict()
            params['weight'] = params['weight'].unsqueeze(2)
            new_layer.load_state_dict(params)
            setattr(model, name, new_layer)
        elif layer_type == 'BatchNorm1d':
            new_layer = nn.BatchNorm2d(l.num_features, l.eps)
            new_layer.load_state_dict(l.state_dict())
            new_layer.eval()
            setattr(model, name, new_layer)
        else:
            convert_to_2d(l)

class QuartzNet(torch.nn.Module):
    def __init__(self, model_config, encoder_weights, decoder_weights):
        super().__init__()
        with open(model_config, 'r') as config:
            model_args = YAML.load(config)
        _ = NeuralModuleFactory(placement=DeviceType.CPU)

        encoder_params = model_args['init_params']['encoder_params']['init_params']
        self.encoder = JasperEncoder(**encoder_params)
        self.encoder.load_state_dict(torch.load(encoder_weights, map_location='cpu'))

        decoder_params = model_args['init_params']['decoder_params']['init_params']
        self.decoder = JasperDecoderForCTC(**decoder_params)
        self.decoder.load_state_dict(torch.load(decoder_weights, map_location='cpu'))

        self.encoder._prepare_for_deployment()
        self.decoder._prepare_for_deployment()
        convert_to_2d(self.encoder)
        convert_to_2d(self.decoder)

    def forward(self, input_signal):
        input_signal = input_signal.unsqueeze(axis=2)
        i_encoded = self.encoder(input_signal)
        i_log_probs = self.decoder(i_encoded)

        shape = i_log_probs.shape
        return i_log_probs.reshape(shape[0], shape[1], shape[3])
