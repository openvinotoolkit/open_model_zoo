"""
 Copyright (C) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging as log
from .utils import InputTransform


class Model:
    '''An abstract model wrapper
    
    An abstract model wrapper can only load model from the disk.
    The ``preprocess`` and ``postprocess`` method should be implemented in concrete class

    Attributes    
        net(CNNNetwork): loaded network
        logger(Logger): instance of the logger
    '''

    def __init__(self, ie, model_path):
        '''Abstract model constructor

        Args:
            ie(openvino.core): instance of Inference Engine core, needs for model loading
            model_path(str, Path): path to model's *.xml file
        '''
        self.logger = log.getLogger()
        self.net = ie.read_network(model_path)
        self.inputs = self.net.input_info
        self.outputs = self.net.outputs
        self.set_batch_size(1)
        self.input_transform = InputTransform()

    def set_inputs_preprocessing(self, reverse_input_channels, mean_values, scale_values):
        self.input_transform = InputTransform(reverse_input_channels, mean_values, scale_values)

    def preprocess(self, inputs):
        '''Interface for preprocess method
        Args:
            inputs: raw input data, data types are defined by concrete model
        Returns:
            - The preprocessed data ready for inference
            - The metadata, which could be used in postprocessing
        '''
        raise NotImplementedError

    def postprocess(self, outputs, meta):
        '''Interface for postrpocess metod
        Args:
            outputs: the model outputs  as dict with `name: tensor` data
            meta: the metadata from the `preprocess` method results
        Returns:
            Postrocessed data in format accoding to conrete model
        '''
        raise NotImplementedError
