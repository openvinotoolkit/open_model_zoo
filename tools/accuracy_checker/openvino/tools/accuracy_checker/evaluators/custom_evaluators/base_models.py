"""
Copyright (c) 2018-2021 Intel Corporation

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
from pathlib import Path
from collections import OrderedDict
import numpy as np

from ...config import ConfigError
from ...utils import get_path, parse_partial_shape
from ...logging import print_info


def create_model(model_config, launcher, launcher_model_mapping, suffix=None, delayed_model_loading=False):
    framework = launcher.config['framework']
    model_class = launcher_model_mapping.get(framework)
    if not model_class:
        raise ValueError('model for framework {} is not supported'.format(framework))
    return model_class(model_config, launcher, suffix, delayed_model_loading)


def create_encoder(model_config, launcher, launcher_model_mapping, delayed_model_loading=False):
    framework = launcher.config['framework']
    if 'predictions' in model_config and not model_config.get('store_predictions', False):
        framework = 'dummy'
    model_class = launcher_model_mapping.get(framework)
    if not model_class:
        raise ValueError('model for framework {} is not supported'.format(framework))
    return model_class(model_config, launcher, 'encoder', delayed_model_loading)


class BaseCascadeModel:
    def __init__(self, network_info, launcher, delayed_model_loading=False):
        self.network_info = network_info
        self.launcher = launcher
        self._part_by_name = None

    def predict(self, identifiers, input_data):
        raise NotImplementedError

    def release(self):
        if self._part_by_name:
            for model in self._part_by_name.values():
                model.release()

    def load_network(self, network_list, launcher):
        for network_dict in network_list:
            self._part_by_name[network_dict['name']].load_network(network_dict['model'], launcher)

    def load_model(self, network_list, launcher):
        for network_dict in network_list:
            self._part_by_name[network_dict['name']].load_model(network_dict, launcher)

    def get_network(self):
        if not self._part_by_name:
            return []
        return [{'name': name, 'model': model.network} for name, model in self._part_by_name.items()]

    def reset(self):
        pass


class BaseDLSDKModel:
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.network_info = network_info
        self.launcher = launcher
        self.default_model_suffix = suffix
        if not hasattr(self, 'output_blob'):
            self.output_blob = None
        if not hasattr(self, 'input_blob'):
            self.input_blob = None
        self.with_prefix = False
        self.is_dynamic = False
        if not delayed_model_loading:
            self.load_model(network_info, launcher, log=True)

    def _reshape_input(self, input_shapes):
        if self.is_dynamic:
            return
        if hasattr(self, 'exec_network') and self.exec_network is not None:
            del self.exec_network
        self.network.reshape(input_shapes)
        self.dynamic_inputs, self.partial_shapes = self.launcher.get_dynamic_inputs(self.network)
        if not self.is_dynamic and self.dynamic_inputs:
            self.exec_network = None
            return
        self.exec_network = self.launcher.ie_core.load_network(self.network, self.launcher.device)

    def load_network(self, network, launcher):
        self.network = network
        self.dynamic_inputs, self.partial_shapes = launcher.get_dynamic_inputs(self.network)
        if self.dynamic_inputs and launcher.dynamic_shapes_policy in ['dynamic', 'default']:
            try:
                self.exec_network = launcher.ie_core.load_network(self.network, launcher.device)
                self.is_dynamic = True
            except RuntimeError as e:
                if launcher.dynamic_shapes_policy == 'dynamic':
                    raise e
                self.is_dynamic = False
                self.exec_network = None
        if not self.dynamic_inputs:
            self.exec_network = launcher.ie_core.load_network(self.network, launcher.device)

    def print_input_output_info(self):
        print_info('{} - Input info:'.format(self.default_model_suffix))
        has_info = hasattr(self.network if self.network is not None else self.exec_network, 'input_info')
        if self.network:
            if has_info:
                network_inputs = OrderedDict(
                    [(name, data.input_data) for name, data in self.network.input_info.items()]
                )
            else:
                network_inputs = self.network.inputs
            network_outputs = self.network.outputs
        else:
            if has_info:
                network_inputs = OrderedDict([
                    (name, data.input_data) for name, data in self.exec_network.input_info.items()
                ])
            else:
                network_inputs = self.exec_network.inputs
            network_outputs = self.exec_network.outputs
        for name, input_info in network_inputs.items():
            print_info('\tLayer name: {}'.format(name))
            print_info('\tprecision: {}'.format(input_info.precision))
            print_info('\tshape {}\n'.format(
                input_info.shape if name not in self.partial_shapes else self.partial_shapes[name]))
        print_info('{} - Output info'.format(self.default_model_suffix))
        for name, output_info in network_outputs.items():
            print_info('\tLayer name: {}'.format(name))
            print_info('\tprecision: {}'.format(output_info.precision))
            print_info('\tshape: {}\n'.format(
                output_info.shape if name not in self.partial_shapes else self.partial_shapes[name]))

    def automatic_model_search(self, network_info):
        model = Path(network_info['model'])
        if model.is_dir():
            is_blob = network_info.get('_model_is_blob')
            if is_blob:
                model_list = list(model.glob('*{}.blob'.format(self.default_model_suffix)))
                if not model_list:
                    model_list = list(model.glob('*.blob'))
            else:
                model_list = list(model.glob('*{}.xml'.format(self.default_model_suffix)))
                blob_list = list(model.glob('*{}.blob'.format(self.default_model_suffix)))
                if not model_list and not blob_list:
                    model_list = list(model.glob('*.xml'))
                    blob_list = list(model.glob('*.blob'))
                    if not model_list:
                        model_list = blob_list
            if not model_list:
                raise ConfigError('Suitable model for {} not found'.format(self.default_model_suffix))
            if len(model_list) > 1:
                raise ConfigError('Several suitable models for {} found'.format(self.default_model_suffix))
            model = model_list[0]
        accepted_suffixes = ['.blob', '.xml', '.onnx']
        if model.suffix not in accepted_suffixes:
            raise ConfigError('Models with following suffixes are allowed: {}'.format(accepted_suffixes))
        print_info('{} - Found model: {}'.format(self.default_model_suffix, model))
        if model.suffix in ['.blob', '.onnx']:
            return model, None
        weights = get_path(network_info.get('weights', model.parent / model.name.replace('xml', 'bin')))
        accepted_weights_suffixes = ['.bin']
        if weights.suffix not in accepted_weights_suffixes:
            raise ConfigError('Weights with following suffixes are allowed: {}'.format(accepted_weights_suffixes))
        print_info('{} - Found weights: {}'.format(self.default_model_suffix, weights))
        return model, weights

    def set_input_and_output(self):
        has_info = hasattr(self.exec_network, 'input_info')
        input_info = self.exec_network.input_info if has_info else self.exec_network.inputs
        input_blob = next(iter(input_info))
        with_prefix = input_blob.startswith(self.default_model_suffix)
        if self.input_blob is None or with_prefix != self.with_prefix:
            if self.output_blob is None:
                output_blob = next(iter(self.exec_network.outputs))
            else:
                output_blob = (
                    '_'.join([self.default_model_suffix, self.output_blob])
                    if with_prefix else self.output_blob.split(self.default_model_suffix + '_')[-1]
                )
            self.input_blob = input_blob
            self.output_blob = output_blob
            self.with_prefix = with_prefix

    def load_model(self, network_info, launcher, log=False):
        if 'onnx_model' in network_info:
            network_info.update(launcher.config)
            model, weights = launcher.convert_model(network_info)
        else:
            model, weights = self.automatic_model_search(network_info)
        if weights is None and model.suffix != '.onnx':
            self.exec_network = launcher.ie_core.import_network(str(model))
        else:
            if weights:
                self.network = launcher.read_network(str(model), str(weights))
            else:
                self.network = launcher.ie_core.read_network(str(model))
            self.load_network(self.network, launcher)
        self.set_input_and_output()
        if log:
            self.print_input_output_info()

    def release(self):
        del self.exec_network
        del self.network
        del self.launcher

    def fit_to_input(self, input_data):
        has_info = hasattr(self.exec_network, 'input_info')
        if has_info:
            input_info = self.exec_network.input_info[self.input_blob].input_data
        else:
            input_info = self.exec_network.inputs[self.input_blob]
        if self.input_blob in self.dynamic_inputs or tuple(input_info.shape) != np.shape(input_data):
            self._reshape_input({self.input_blob: np.shape(input_data)})

        return {self.input_blob: np.array(input_data)}

    def predict(self, identifiers, input_data):
        raise NotImplementedError


class BaseOpenVINOModel(BaseDLSDKModel):
    def _reshape_input(self, input_shapes):
        if self.is_dynamic:
            return
        if hasattr(self, 'exec_network') and self.exec_network is not None:
            del self.infer_request
            del self.exec_network
        self.launcher.reshape_network(self.network, input_shapes)
        self.dynamic_inputs, self.partial_shapes = self.launcher.get_dynamic_inputs(self.network)
        if not self.is_dynamic and self.dynamic_inputs:
            self.exec_network = None
            return
        self.exec_network = self.launcher.ie_core.compile_model(self.network, self.launcher.device)
        self.infer_request = None

    def predict(self, identifiers, input_data):
        raise NotImplementedError

    def load_network(self, network, launcher):
        self.infer_request = None
        self.network = network
        self.dynamic_inputs, self.partial_shapes = launcher.get_dynamic_inputs(self.network)
        if self.dynamic_inputs and launcher.dynamic_shapes_policy in ['dynamic', 'default']:
            try:
                self.exec_network = launcher.ie_core.compile_model(self.network, launcher.device)
                self.is_dynamic = True
            except RuntimeError as e:
                if launcher.dynamic_shapes_policy == 'dynamic':
                    raise e
                self.is_dynamic = False
                self.exec_network = None
        if not self.dynamic_inputs:
            self.exec_network = launcher.ie_core.compile_model(self.network, launcher.device)

    def load_model(self, network_info, launcher, log=False):
        if 'onnx_model' in network_info:
            network_info.update(launcher.config)
            model, weights = launcher.convert_model(network_info)
        else:
            model, weights = self.automatic_model_search(network_info)
        if weights is None and model.suffix != '.onnx':
            self.exec_network = launcher.ie_core.import_network(str(model))
        else:
            if weights:
                self.network = launcher.read_network(str(model), str(weights))
            else:
                self.network = launcher.ie_core.read_network(str(model))
            self.load_network(self.network, launcher)
        self.set_input_and_output()
        if log:
            self.print_input_output_info()

    def print_input_output_info(self):
        self.launcher.print_input_output_info(
            self.network if self.network is not None else self.exec_network, self.default_model_suffix)

    def set_input_and_output(self):
        inputs = self.network.inputs if self.network is not None else self.exec_network.inputs
        outputs = self.network.outputs if self.network is not None else self.exec_network.outputs
        input_blob = next(iter(inputs)).get_node().friendly_name
        with_prefix = input_blob.startswith(self.default_model_suffix)
        if self.input_blob is None or with_prefix != self.with_prefix:
            if self.output_blob is None:
                output_blob = next(iter(outputs)).get_node().friendly_name
            else:
                output_blob = (
                    '_'.join([self.default_model_suffix, self.output_blob])
                    if with_prefix else self.output_blob.split(self.default_model_suffix + '_')[-1]
                )
            self.input_blob = input_blob
            self.output_blob = output_blob
            self.with_prefix = with_prefix

    @property
    def inputs(self):
        if self.network:
            return {node.get_node().friendly_name: node.get_node() for node in self.network.inputs}
        return {node.get_node().friendly_name: node.get_node() for node in self.exec_network.inputs}

    def fit_to_input(self, input_data):
        input_info = self.inputs[self.input_blob]
        if (self.input_blob in self.dynamic_inputs or
            parse_partial_shape(input_info.get_partial_shape()) != np.shape(input_data)):
            self._reshape_input({self.input_blob: np.shape(input_data)})

        return {self.input_blob: np.array(input_data)}

    def infer(self, input_data):
        if self.infer_request is None:
            self.infer_request = self.exec_network.create_infer_request()
        outputs = self.infer_request.infer(input_data)
        return {
            out_node.get_node().friendly_name: out_res
            for out_node, out_res in zip(self.exec_network.outputs, outputs)
        }


class BaseONNXModel:
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.network_info = network_info
        self.launcher = launcher
        self.default_model_suffix = suffix
        if not delayed_model_loading:
            model = self.automatic_model_search(network_info)
            self.inference_session = launcher.create_inference_session(str(model))
            self.input_blob = next(iter(self.inference_session.get_inputs()))
            self.output_blob = next(iter(self.inference_session.get_outputs()))

    def fit_to_input(self, input_data):
        return {self.input_blob.name: input_data}

    def release(self):
        del self.inference_session

    def automatic_model_search(self, network_info):
        model = Path(network_info['model'])
        if model.is_dir():
            model_list = list(model.glob('*{}.onnx'.format(self.default_model_suffix)))
            if not model_list:
                model_list = list(model.glob('*.onnx'))
            if not model_list:
                raise ConfigError('Suitable model for {} not found'.format(self.default_model_suffix))
            if len(model_list) > 1:
                raise ConfigError('Several suitable models for {} found'.format(self.default_model_suffix))
            model = model_list[0]
        accepted_suffixes = ['.onnx']
        if model.suffix not in accepted_suffixes:
            raise ConfigError('Models with following suffixes are allowed: {}'.format(accepted_suffixes))
        print_info('{} - Found model: {}'.format(self.default_model_suffix, model))
        return model


class BaseOpenCVModel:
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.network_info = network_info
        self.launcher = launcher
        self.default_model_suffix = suffix
        if not delayed_model_loading:
            self.network = launcher.create_network(network_info['model'], network_info.get('weights', ''))
            network_info.update(launcher.config)
            input_shapes = launcher.get_inputs_from_config(network_info)
            self.input_blob = next(iter(input_shapes))
            self.input_shape = input_shapes[self.input_blob]
            self.network.setInputsNames(list(self.input_blob))
            self.output_blob = next(iter(self.network.getUnconnectedOutLayersNames()))

    def fit_to_input(self, input_data):
        return {self.input_blob: input_data.astype(np.float32)}

    def release(self):
        del self.network


class BaseTFModel:
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.network_info = network_info
        self.launcher = launcher
        self.default_model_suffix = suffix
        if not delayed_model_loading:
            model = self.automatic_model_search(network_info)
            self.inference_session = launcher.create_inference_session(str(model))

    def fit_to_input(self, input_data):
        raise NotImplementedError

    def predict(self, identifiers, input_data):
        raise NotImplementedError

    def release(self):
        del self.inference_session

    @staticmethod
    def automatic_model_search(network_info):
        model = Path(network_info['model'])
        return model
