"""
 Copyright (C) 2020-2022 Intel Corporation

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


class WrapperError(RuntimeError):
    def __init__(self, wrapper_name, message):
        self.message = f"{wrapper_name}: {message}"
        super().__init__(self.message)


class Model:
    '''An abstract model wrapper

    An abstract model wrapper can only load model from the disk.
    The ``preprocess`` and ``postprocess`` methods should be implemented in a concrete class

    Attributes:
        model_adapter(ModelAdapter): allows working with the specified executor
        logger(Logger): instance of the logger
    '''

    __model__ = None # Abstract wrapper has no name

    def __init__(self, model_adapter, configuration=None, preload=False):
        '''Model constructor

        Args:
            model_adapter(ModelAdapter): allows working with the specified executor
        '''
        self.logger = log.getLogger()
        self.model_adapter = model_adapter
        self.inputs = self.model_adapter.get_input_layers()
        self.outputs = self.model_adapter.get_output_layers()
        for name, parameter in self.parameters().items():
            self.__setattr__(name, parameter.default_value)
        self.load_config(configuration if configuration else {})
        self.model_loaded = False
        if preload:
            self.load()

    @classmethod
    def get_model(cls, name):
        subclasses = [subclass for subclass in cls.get_subclasses() if subclass.__model__]
        if cls.__model__:
            subclasses.append(cls)
        for subclass in subclasses:
            if name.lower() == subclass.__model__.lower():
                return subclass
        raise WrapperError(cls.__model__, 'There is no model with name "{}" in list: {}'.
                         format(name, ', '.join([subclass.__model__ for subclass in subclasses])))

    @classmethod
    def create_model(cls, name, model_adapter, configuration=None, preload=False):
        Model = cls.get_model(name)
        return Model(model_adapter, configuration, preload)

    @classmethod
    def get_subclasses(cls):
        all_subclasses = []
        for subclass in cls.__subclasses__():
            all_subclasses.append(subclass)
            all_subclasses.extend(subclass.get_subclasses())
        return all_subclasses

    @classmethod
    def available_wrappers(cls):
        available_classes = [cls] if cls.__model__ else []
        available_classes.extend(cls.get_subclasses())
        return [subclass.__model__ for subclass in available_classes if subclass.__model__]

    @classmethod
    def parameters(cls):
        parameters = {}
        return parameters

    def load_config(self, config):
        parameters = self.parameters()
        for name, value in config.items():
            if name in parameters:
                errors = parameters[name].validate(value)
                if errors:
                    log.error(f'Error with "{name}" parameter:')
                    for error in errors:
                        log.error(f"\t{error}")
                    raise WrapperError(self.__model__, 'Incorrect user configuration')
                value = parameters[name].get_value(value)
                self.__setattr__(name, value)
            else:
                log.warning(f'The parameter "{name}" not found in {self.__model__} wrapper, will be omitted')

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
                raise WrapperError(self.__model__, "Expected {} input blob{}, but {} found: {}".format(
                    number_of_inputs, 's' if number_of_inputs !=1 else '',
                    len(self.inputs), ', '.join(self.inputs)
                ))
        else:
            if not len(self.inputs) in number_of_inputs:
                raise WrapperError(self.__model__, "Expected {} or {} input blobs, but {} found: {}".format(
                    ', '.join(str(n) for n in number_of_inputs[:-1]), int(number_of_inputs[-1]),
                    len(self.inputs), ', '.join(self.inputs)
                ))

        if not isinstance(number_of_outputs, tuple):
            if len(self.outputs) != number_of_outputs and number_of_outputs != -1:
                raise WrapperError(self.__model__, "Expected {} output blob{}, but {} found: {}".format(
                    number_of_outputs, 's' if number_of_outputs !=1 else '',
                    len(self.outputs), ', '.join(self.outputs)
                ))
        else:
            if not len(self.outputs) in number_of_outputs:
                raise WrapperError(self.__model__, "Expected {} or {} output blobs, but {} found: {}".format(
                    ', '.join(str(n) for n in number_of_outputs[:-1]), int(number_of_outputs[-1]),
                    len(self.outputs), ', '.join(self.outputs)
                ))

    def __call__(self, input_data):
        '''
        Applies the preprocessing, synchronous inference and postprocessing method of model wrapper
        '''
        dict_data, input_meta = self.preprocess(input_data)
        raw_result = self.infer_sync(dict_data)
        return self.postprocess(raw_result, input_meta), input_meta

    def load(self, force=False):
        if not self.model_loaded or force:
            self.model_loaded = True
            self.model_adapter.load_model()

    def reshape(self, new_shape):
        if self.model_loaded:
            log.warning(f'{self.__model__}: the model already loaded to device, should be reloaded after reshaping.')
            self.model_loaded = False
        self.model_adapter.reshape_model(new_shape)
        self.inputs = self.model_adapter.get_input_layers()
        self.outputs = self.model_adapter.get_output_layers()

    def infer_sync(self, dict_data):
        return self.model_adapter.infer_sync(dict_data)

    def infer_async(self, dict_data, callback_data):
        self.model_adapter.infer_async(dict_data, callback_data)

    def is_ready(self):
        return self.model_adapter.is_ready()

    def await_all(self):
        self.model_adapter.await_all()

    def await_any(self):
        self.model_adapter.await_any()

    def log_layers_info(self):
        for name, metadata in self.inputs.items():
            log.info('\tInput layer: {}, shape: {}, precision: {}, layout: {}'.format(name, metadata.shape,
                                                                                      metadata.precision, metadata.layout))
        for name, metadata in self.outputs.items():
            log.info('\tOutput layer: {}, shape: {}, precision: {}, layout: {}'.format(name, metadata.shape,
                                                                                       metadata.precision, metadata.layout))
