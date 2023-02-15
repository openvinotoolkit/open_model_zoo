"""
 Copyright (c) 2021-2023 Intel Corporation

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
from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class Metadata:
    names: Set[str] = field(default_factory=set)
    shape: List[int] = field(default_factory=list)
    layout: str = ''
    precision: str = ''
    type: str = ''
    meta: Dict = field(default_factory=dict)


class ModelAdapter(metaclass=abc.ABCMeta):
    '''
    An abstract Model Adapter with the following interface:

        - Reading the model from disk or other place
        - Loading the model to the device
        - Accessing the information about inputs/outputs
        - The model reshaping
        - Synchronous model inference
        - Asynchronous model inference
    '''
    precisions = ('FP32', 'I32', 'FP16', 'I16', 'I8', 'U8')

    @abc.abstractmethod
    def __init__(self):
        '''
        An abstract Model Adapter constructor.
        Reads the model from disk or other place.
        '''

    @abc.abstractmethod
    def load_model(self):
        '''
        Loads the model on the device.
        '''

    @abc.abstractmethod
    def get_input_layers(self):
        '''
        Gets the names of model inputs and for each one creates the Metadata structure,
           which contains the information about the input shape, layout, precision
           in OpenVINO format, meta (optional)

        Returns:
            - the dict containing Metadata for all inputs
        '''

    @abc.abstractmethod
    def get_output_layers(self):
        '''
        Gets the names of model outputs and for each one creates the Metadata structure,
           which contains the information about the output shape, layout, precision
           in OpenVINO format, meta (optional)

        Returns:
            - the dict containing Metadata for all outputs
        '''

    @abc.abstractmethod
    def reshape_model(self, new_shape):
        '''
        Reshapes the model inputs to fit the new input shape.

        Args:
            - new_shape (dict): the dictionary with inputs names as keys and
                list of new shape as values in the following format:
                {
                    'input_layer_name_1': [1, 128, 128, 3],
                    'input_layer_name_2': [1, 128, 128, 3],
                    ...
                }
        '''

    @abc.abstractmethod
    def infer_sync(self, dict_data):
        '''
        Performs the synchronous model inference. The infer is a blocking method.

        Args:
            - dict_data: it's submitted to the model for inference and has the following format:
                {
                    'input_layer_name_1': data_1,
                    'input_layer_name_2': data_2,
                    ...
                }

        Returns:
            - raw result (dict) - model raw output in the following format:
                {
                    'output_layer_name_1': raw_result_1,
                    'output_layer_name_2': raw_result_2,
                    ...
                }
        '''

    @abc.abstractmethod
    def infer_async(self, dict_data, callback_fn, callback_data):
        '''
        Performs the asynchronous model inference and sets
        the callback for inference completion. Also, it should
        define get_raw_result() function, which handles the result
        of inference from the model.

        Args:
            - dict_data: it's submitted to the model for inference and has the following format:
                {
                    'input_layer_name_1': data_1,
                    'input_layer_name_2': data_2,
                    ...
                }
            - callback_fn: the callback function, which is defined outside the adapter
            - callback_data: the data for callback, that will be taken after the model inference is ended
        '''

    @abc.abstractmethod
    def is_ready(self):
        '''
        In case of asynchronous execution checks if one can submit input data
        to the model for inference, or all infer requests are busy.

        Returns:
            - the boolean flag whether the input data can be
                submitted to the model for inference or not
        '''

    @abc.abstractmethod
    def await_all(self):
        '''
        In case of asynchronous execution waits the completion of all
        busy infer requests.
        '''

    @abc.abstractmethod
    def await_any(self):
        '''
        In case of asynchronous execution waits the completion of any
        busy infer request until it becomes available for the data submission.
        '''
