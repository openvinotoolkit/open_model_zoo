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

    Attributes:
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

    def _check_io_number(self, number_of_inputs, number_of_outputs):
        '''Checking actual number of input/output blobs with supported by model wrapper
        

        Args:
            number_of_inputs(int, Tuple(int)): number of input blobs, supported by wrapper. Use -1 to omit check
            number_of_outputs(int, Tuple(int)): number of output blobs, supported by wrapper. Use -1 to omit check
        
        Raises:
            RuntimeError: if loaded model has unsupported number of input or output blob
        '''
        if not isinstance(number_of_inputs, tuple):
            if len(self.inputs) != number_of_inputs and number_of_inputs != -1:
                raise RuntimeError("Expected {} input blob{}, but {} found: {}".format(
                    number_of_inputs, 's' if number_of_inputs !=1 else '', len(self.inputs), ', '.join(self.inputs)
                ))
        else:
            if not len(self.inputs) in number_of_inputs:
                raise RuntimeError("Expected {} or {} input blobs, but {} found: {}".format(
                    ', '.join(str(n) for n in number_of_inputs[:-1]), int(number_of_inputs[-1]), len(self.inputs), 
                    ', '.join(self.inputs)
                ))

        if not isinstance(number_of_outputs, tuple):
            if len(self.outputs) != number_of_outputs and number_of_outputs != -1:
                raise RuntimeError("Expected {} output blob{}, but {} found: {}".format(
                    number_of_outputs, 's' if number_of_outputs !=1 else '', len(self.outputs), ', '.join(self.outputs)
                ))
        else:
            if not len(self.outputs) in number_of_outputs:
                raise RuntimeError("Expected {} or {} output blobs, but {} found: {}".format(
                    ', '.join(str(n) for n in number_of_outputs[:-1]), int(number_of_outputs[-1]), len(self.outputs),
                    ', '.join(self.outputs)
                ))
