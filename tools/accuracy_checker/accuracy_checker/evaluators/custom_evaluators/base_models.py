"""
Copyright (c) 2018-2024 Intel Corporation

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
from ...utils import get_path, parse_partial_shape, contains_any
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
        if len(self._part_by_name) == 1 and 'name' not in network_list[0]:
            next(iter(self._part_by_name.values())).load_model(network_list[0]['model'], launcher)
            return
        for network_dict in network_list:
            self._part_by_name[network_dict['name']].load_network(network_dict['model'], launcher)

    def load_model(self, network_list, launcher):
        if len(self._part_by_name) == 1 and 'name' not in network_list[0]:
            next(iter(self._part_by_name.values())).load_model(network_list[0], launcher)
            return
        for network_dict in network_list:
            self._part_by_name[network_dict['name']].load_model(network_dict, launcher)

    def get_network(self):
        if not self._part_by_name:
            return []
        return [{'name': name, 'model': model.network} for name, model in self._part_by_name.items()]

    def reset(self):
        pass

    @staticmethod
    def fill_part_with_model(network_info, parts, models_args, is_blob, delayed_model_loading):
        if models_args and not delayed_model_loading:
            for idx, part in enumerate(parts):
                part_info = network_info.get(part, {})
                if not contains_any(part_info, ['model', 'onnx_model']) and models_args:
                    part_info['model'] = models_args[idx if len(models_args) > idx else 0]
                    part_info['_model_is_blob'] = is_blob
                network_info.update({part: part_info})
        return network_info

    @staticmethod
    def automatic_model_search(network_info):
        model = Path(network_info['model'])
        model_name = network_info["name"]
        if model.is_dir():
            is_blob = network_info.get('_model_is_blob')
            if is_blob:
                model_list = list(model.glob('*{}.blob'.format(model_name)))
                if not model_list:
                    model_list = list(model.glob('*.blob'))
            else:
                model_list = list(model.glob('*{}*.xml'.format(model_name)))
                blob_list = list(model.glob('*{}*.blob'.format(model_name)))
                onnx_list = list(model.glob('*{}*.onnx'.format(model_name)))
                if not model_list and not blob_list and not onnx_list:
                    model_list = list(model.glob('*.xml'))
                    blob_list = list(model.glob('*.blob'))
                    onnx_list = list(model.glob('*.onnx'))
                if not model_list:
                    model_list = blob_list if blob_list else onnx_list
            if not model_list:
                raise ConfigError('Suitable model for {} not found'.format(model_name))
            if len(model_list) > 1:
                raise ConfigError('Several suitable models for {} found'.format(model_name))
            model = model_list[0]
        accepted_suffixes = ['.xml', '.onnx']
        if model.suffix not in accepted_suffixes:
            raise ConfigError('Models with following suffixes are allowed: {}'.format(accepted_suffixes))
        print_info('{} - Found model: {}'.format(model_name, model))
        if model.suffix in ['.blob', '.onnx']:
            return model, None
        weights = get_path(network_info.get('weights', model.parent / model.name.replace('xml', 'bin')))
        accepted_weights_suffixes = ['.bin']
        if weights.suffix not in accepted_weights_suffixes:
            raise ConfigError('Weights with following suffixes are allowed: {}'.format(accepted_weights_suffixes))
        print_info('{} - Found weights: {}'.format(model_name, weights))
        return model, weights


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
                model_list = list(model.glob('*{}*.xml'.format(self.default_model_suffix)))
                blob_list = list(model.glob('*{}*.blob'.format(self.default_model_suffix)))
                onnx_list = list(model.glob('*{}*.onnx'.format(self.default_model_suffix)))
                if not model_list and not blob_list and not onnx_list:
                    model_list = list(model.glob('*.xml'))
                    blob_list = list(model.glob('*.blob'))
                    onnx_list = list(model.glob('*.onnx'))
                if not model_list:
                    model_list = blob_list if blob_list else onnx_list
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
        network = self.exec_network if self.exec_network is not None else self.network
        has_info = hasattr(network, 'input_info')
        input_info = network.input_info if has_info else network.inputs
        input_blob = next(iter(input_info))
        with_prefix = input_blob.startswith(self.default_model_suffix)
        if self.input_blob is None or with_prefix != self.with_prefix:
            if self.output_blob is None:
                output_blob = next(iter(network.outputs))
            else:
                output_blob = (
                    '_'.join([self.default_model_suffix, self.output_blob])
                    if with_prefix else self.output_blob.split(self.default_model_suffix + '_')[-1]
                )
            self.input_blob = input_blob
            self.output_blob = output_blob
            self.with_prefix = with_prefix
            if hasattr(self, 'adapter') and self.adapter is not None:
                self.adapter.output_blob = output_blob

    def load_model(self, network_info, launcher, log=False):
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
    def input_tensors_mapping(self):
        inputs = self.network.inputs if self.network is not None else self.exec_network.inputs
        node_to_tensor = {}
        for idx, input_desc in enumerate(inputs):
            tensor_names = input_desc.get_tensor().get_names()
            node_to_tensor[input_desc.get_node().friendly_name] = idx if not tensor_names else next(iter(tensor_names))

        return node_to_tensor

    def input_index_mapping(self):
        inputs = self.network.inputs if self.network is not None else self.exec_network.inputs
        return {inp.get_node().friendly_name: idx for idx, inp in enumerate(inputs)}

    def _reshape_input(self, input_shapes):
        if self.is_dynamic:
            return
        if hasattr(self, 'exec_network') and self.exec_network is not None:
            if hasattr(self, 'infer_request'):
                del self.infer_request
            del self.exec_network
        index_mapping = self.input_index_mapping()
        input_shapes_for_tensors = {index_mapping[name]: shape for name, shape in input_shapes.items()}
        self.launcher.reshape_network(self.network, input_shapes_for_tensors)
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
        model, weights = self.automatic_model_search(network_info)
        if weights is None and model.suffix != '.onnx':
            self.exec_network = launcher.ie_core.import_network(str(model))
        else:
            if weights:
                self.network = launcher.read_network(str(model), str(weights))
            else:
                self.network = launcher.read_network(str(model), None)
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
                self.output_blob = next(iter(outputs)).get_node().friendly_name
            self.input_blob = input_blob
            self.with_prefix = with_prefix
            if hasattr(self, 'adapter') and self.adapter is not None:
                self.adapter.output_blob = self.output_blob

    @property
    def inputs(self):
        if self.network:
            return {node.get_node().friendly_name: node.get_node() for node in self.network.inputs}
        return {node.get_node().friendly_name: node.get_node() for node in self.exec_network.inputs}

    @property
    def outputs(self):
        if self.network:
            return {node.get_node().friendly_name: node.get_node() for node in self.network.outputs}
        return {node.get_node().friendly_name: node.get_node() for node in self.exec_network.outputs}

    @property
    def additional_output_mapping(self):
        out_tensor_name_to_node = {}
        for out in self.network.outputs:
            if not out.names:
                continue
            for name in out.names:
                out_tensor_name_to_node[name] = out.get_node().friendly_name
        return out_tensor_name_to_node

    def fit_to_input(self, input_data):
        input_info = self.inputs[self.input_blob]
        if (self.input_blob in self.dynamic_inputs or
            parse_partial_shape(input_info.get_partial_shape()) != np.shape(input_data)):
            self._reshape_input({self.input_blob: np.shape(input_data)})

        return {self.input_blob: np.array(input_data)}

    def infer(self, input_data, raw_results=False):
        if not hasattr(self, 'infer_request') or self.infer_request is None:
            self.infer_request = self.exec_network.create_infer_request()
        tensors_mapping = self.input_tensors_mapping()
        feed_dict = {tensors_mapping[name]: data for name, data in input_data.items()}
        outputs = self.infer_request.infer(feed_dict)
        res_outputs = {out_node.get_node().friendly_name: out_res for out_node, out_res in outputs.items()}
        if raw_results:
            return res_outputs, outputs
        return res_outputs


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
            model_list = list(model.glob('*{}*.onnx'.format(self.default_model_suffix)))
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


class BaseCaffeModel:
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.network_info = network_info
        self.launcher = launcher
        self.default_model_suffix = suffix

    def fit_to_input(self, data, layer_name, layout, precision, tmpl=None):
        return self.launcher.fit_to_input(data, layer_name, layout, precision, template=tmpl)

    def predict(self, identifiers, input_data):
        raise NotImplementedError

    def release(self):
        del self.net

    def automatic_model_search(self, network_info):
        model = Path(network_info.get('model', ''))
        weights = network_info.get('weights')
        if model.is_dir():
            models_list = list(Path(model).glob('{}.prototxt'.format(self.default_model_name)))
            if not models_list:
                models_list = list(Path(model).glob('*.prototxt'))
            if not models_list:
                raise ConfigError('Suitable model description is not detected')
            if len(models_list) != 1:
                raise ConfigError('Several suitable models found, please specify required model')
            model = models_list[0]
        if weights is None or Path(weights).is_dir():
            weights_dir = weights or model.parent
            weights = Path(weights_dir) / model.name.replace('prototxt', 'caffemodel')
            if not weights.exists():
                weights_list = list(weights_dir.glob('*.caffemodel'))
                if not weights_list:
                    raise ConfigError('Suitable weights is not detected')
                if len(weights_list) != 1:
                    raise ConfigError('Several suitable weights found, please specify required explicitly')
                weights = weights_list[0]
        weights = Path(weights)
        accepted_suffixes = ['.prototxt']
        if model.suffix not in accepted_suffixes:
            raise ConfigError('Models with following suffixes are allowed: {}'.format(accepted_suffixes))
        print_info('{} - Found model: {}'.format(self.default_model_name, model))
        accepted_weights_suffixes = ['.caffemodel']
        if weights.suffix not in accepted_weights_suffixes:
            raise ConfigError('Weights with following suffixes are allowed: {}'.format(accepted_weights_suffixes))
        print_info('{} - Found weights: {}'.format(self.default_model_name, weights))
        return model, weights
