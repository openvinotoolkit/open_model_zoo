"""
 Copyright (c) 2021 Intel Corporation

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

import abc


class ModelAdapter(metaclass=abc.ABCMeta):
    '''
    An abstract Model Adapter with the following interface:

        - Loading the model with the specific executor
        - Accessing the information about input/output layers
        - Synchronous model inference
        - The model reshaping
    '''

    @abc.abstractmethod
    def __init__(self, model_name, config, *args):
        '''An abstract Model Adapter constructor

        Args:
            model_name: specify the path / name of the model to load
            config(dict): specify the config for the executor
            *args: other arguments for loading the model depending from the adapter
        '''
        self.load_model(model_name, config, *args)

    @abc.abstractmethod
    def load_model(self, model_name, config, *args):
        '''
        Loading the model with the specific executor.

        Args:
            model_name: specify the path / name of the model to load
            config(dict): specify the config for the executor
            *args: other arguments for loading the model depending from the adapter
        '''

    @abc.abstractmethod
    def get_input_layers(self):
        '''
        Returns:
            - the list with names of model input layers
        '''

    @abc.abstractmethod
    def get_output_layers(self):
        '''
        Returns:
            - the list with names of model output layers
        '''

    @abc.abstractmethod
    def get_input_layer_shape(self, input_layer_name):
        '''
        Returns the shape of specified input layer by its name.

        Args:
            input_layer_name(str): name of input layer to refer

        Returns:
            - the list represented the input 4D layer shape in NWHC or NCHW layout
        '''

    @abc.abstractmethod
    def get_output_layer_shape(self, output_layer_name):
        '''
        Returns the shape of specified output layer by its name.

        Args:
            output_layer_name(str): name of output layer to refer

        Returns:
            - the list represented the output 4D layer shape
        '''

    @abc.abstractmethod
    def get_input_layer_precision(self, input_layer_name):
        '''
        Returns the precision of specified input layer by its name.

        Args:
            input_layer_name(str): name of input layer to refer

        Returns:
            - the string representation of input layer precision
        '''

    @abc.abstractmethod
    def get_output_layer_precision(self, output_layer_name):
        '''
        Returns the precision of specified output layer by its name.

        Args:
            output_layer_name(str): name of output layer to refer

        Returns:
            - the string representation of output layer precision
        '''

    @abc.abstractmethod
    def create_infer_request(self, input_layer_name, data):
        '''
        Forms the infer request for infer method.
        It's called in Model Wrapper preprocess() method, after the preprocessing operations are done.

        Args:
            input_layer_name(str): name of input layer
            data: preprocessed numpy array tensor

        Returns:
            - the infer request - it might be a dict or a custom structure for some executors
        '''

    @abc.abstractmethod
    def infer(self, infer_request):
        '''
        Performs the synchronous model inference. The infer is a blocking method.

        Args:
            - infer_request: contains the data and is submitted to the model for inference

        Returns:
            - the raw result inferred by the model
        '''

    @abc.abstractmethod
    def reshape_model(self, new_shape):
        '''
        Reshapes the model to fit the new input shape.

        Args:
            - new_shape(dict): input_layer(str): new_shape(list) }
        '''
