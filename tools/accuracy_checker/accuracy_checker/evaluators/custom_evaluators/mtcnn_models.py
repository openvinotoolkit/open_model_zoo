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

import pickle  # nosec B403  # disable import-pickle check
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ...preprocessor import PreprocessingExecutor
from ...config import ConfigError
from ...utils import (
    contains_any, extract_image_representations, read_pickle, get_path, parse_partial_shape, postprocess_output_name
)
from .mtcnn_evaluator_utils import cut_roi, calibrate_predictions, nms, transform_for_callback
from ...logging import print_info
from ...launcher import InputFeeder
from .base_models import BaseOpenVINOModel
from ...adapters import create_adapter


def build_stages(models_info, preprocessors_config, launcher, model_args, delayed_model_loading=False):
    required_stages = ['pnet']
    stages_mapping = OrderedDict([
        ('pnet', {
            'caffe': CaffeProposalStage, 'dlsdk': DLSDKProposalStage,
            'dummy': DummyProposalStage, 'openvino': OpenVINOProposalStage}),
        ('rnet', {'caffe': CaffeRefineStage, 'dlsdk': DLSDKRefineStage,
                  'openvino': OpenVINORefineStage}),
        ('onet', {'caffe': CaffeOutputStage, 'dlsdk': DLSDKOutputStage, 'openvino': OpenVINOOutputStage})
    ])
    framework = launcher.config['framework']
    common_preprocessor = PreprocessingExecutor(preprocessors_config)
    stages = OrderedDict()
    for stage_name, stage_classes in stages_mapping.items():
        if stage_name not in models_info:
            if stage_name not in required_stages:
                continue
            raise ConfigError('{} required for evaluation'.format(stage_name))
        model_config = models_info[stage_name]
        if 'predictions' in model_config and not model_config.get('store_predictions', False):
            stage_framework = 'dummy'
        else:
            stage_framework = framework
        if not delayed_model_loading:
            if not contains_any(model_config, ['model', 'caffe_model']) and stage_framework != 'dummy':
                if model_args:
                    model_config['model'] = model_args[len(stages) if len(model_args) > 1 else 0]
        stage = stage_classes.get(stage_framework)
        if not stage_classes:
            raise ConfigError('{} stage does not support {} framework'.format(stage_name, stage_framework))
        stage_preprocess = models_info[stage_name].get('preprocessing', [])
        model_specific_preprocessor = PreprocessingExecutor(stage_preprocess)
        stages[stage_name] = stage(
            models_info[stage_name], model_specific_preprocessor, common_preprocessor, launcher, delayed_model_loading
        )

    if not stages:
        raise ConfigError('please provide information about MTCNN pipeline stages')

    return stages


class BaseStage:
    def __init__(self, model_info, model_specific_preprocessor, common_preprocessor, delayed_model_loading=False):
        self.model_info = model_info
        self.model_specific_preprocessor = model_specific_preprocessor
        self.common_preprocessor = common_preprocessor
        self.input_feeder = None
        self.store = model_info.get('store_predictions', False)
        self.predictions = []

    def predict(self, input_blobs, batch_meta, output_callback=None):
        raise NotImplementedError

    def preprocess_data(self, batch_input, batch_annotation, previous_stage_prediction, *args, **kwargs):
        raise NotImplementedError

    def postprocess_result(self, identifiers, this_stage_result, batch_meta, previous_stage_result, *args, **kwargs):
        raise NotImplementedError

    def release(self):
        pass

    def reset(self):
        self._predictions = []

    def dump_predictions(self):
        if not hasattr(self, 'prediction_file'):
            prediction_file = Path(self.model_info.get('predictions', 'predictions.pickle'))
            self.prediction_file = prediction_file
        with self.prediction_file.open('wb') as out_file:
            pickle.dump(self._predictions, out_file)

    def update_preprocessing(self, preprocessor):
        self.common_preprocessor = preprocessor


class ProposalBaseStage(BaseStage):
    default_model_name = 'mtcnn-p'
    default_model_suffix = 'pnet'

    def __init__(self, model_info, model_specific_preprocessor, common_preprocessor, delayed_model_loading=False):
        super().__init__(model_info, model_specific_preprocessor, common_preprocessor)
        self.adapter = None
        self.input_feeder = None
        self._predictions = []

    def preprocess_data(self, batch_input, batch_annotation, *args, **kwargs):
        batch_input = self.model_specific_preprocessor.process(batch_input, batch_annotation)
        batch_input = self.common_preprocessor.process(batch_input, batch_annotation)
        _, batch_meta = extract_image_representations(batch_input)
        filled_inputs = self.input_feeder.fill_inputs(batch_input) if self.input_feeder else batch_input
        return filled_inputs, batch_meta

    def postprocess_result(self, identifiers, this_stage_result, batch_meta, *args, **kwargs):
        result = self.adapter.process(this_stage_result, identifiers, batch_meta) if self.adapter else this_stage_result
        if self.store:
            self._predictions.extend(result)
        return result

    def predict(self, input_blobs, batch_meta, output_callback=None):
        return self._infer(input_blobs, batch_meta)

    def dump_predictions(self):
        if not hasattr(self, 'prediction_file'):
            prediction_file = Path(self.model_info.get('predictions', 'pnet_predictions.pickle'))
            self.prediction_file = prediction_file
        with self.prediction_file.open('wb') as out_file:
            pickle.dump(self._predictions, out_file)


class DummyProposalStage(ProposalBaseStage):
    def __init__(self, model_info, model_specific_preprocessor, common_preprocessor, *args, **kwargs):
        super().__init__(model_info, model_specific_preprocessor, common_preprocessor)
        self._index = 0
        if 'predictions' not in self.model_info:
            raise ConfigError('predictions_file is not found')
        self._predictions = read_pickle(self.model_info['predictions'])
        self.iterator = 0

    def preprocess_data(self, batch_input, batch_annotation, *args, **kwargs):
        _, batch_meta = extract_image_representations(batch_input)
        return batch_input, batch_meta

    def _infer(self, input_blobs, batch_meta):
        batch_size = len(batch_meta)
        results = self._predictions[self._index:self._index + batch_size]
        self._index += batch_size
        return results

    def postprocess_result(self, identifiers, this_stage_result, batch_meta, *args, **kwargs):
        return this_stage_result


class RefineBaseStage(BaseStage):
    input_size = 24
    include_boundaries = True
    default_model_name = 'mtcnn-r'

    def preprocess_data(self, batch_input, batch_annotation, previous_stage_prediction, *args, **kwargs):
        batch_input = self.model_specific_preprocessor.process(batch_input, batch_annotation)
        batch_input = self.common_preprocessor.process(batch_input, batch_annotation)
        _, batch_meta = extract_image_representations(batch_input)
        batch_input = [
            cut_roi(input_image, prediction, self.input_size, include_bound=self.include_boundaries)
            for input_image, prediction in zip(batch_input, previous_stage_prediction)
        ]
        filled_inputs = self.input_feeder.fill_inputs(batch_input) if self.input_feeder else batch_input
        return filled_inputs, batch_meta

    def postprocess_result(self, identifiers, this_stage_result, batch_meta, previous_stage_result, *args, **kwargs):
        result = calibrate_predictions(
            previous_stage_result, this_stage_result, 0.7, self.model_info['outputs'], 'Union'
        )
        if self.store:
            self._predictions.extend(result)
        return result

    def predict(self, input_blobs, batch_meta, output_callback=None):
        return self._infer(input_blobs, batch_meta)

    def dump_predictions(self):
        if not hasattr(self, 'prediction_file'):
            prediction_file = Path(self.model_info.get('predictions', 'rnet_predictions.pickle'))
            self.prediction_file = prediction_file
        with self.prediction_file.open('wb') as out_file:
            pickle.dump(self._predictions, out_file)


class OutputBaseStage(RefineBaseStage):
    input_size = 48
    include_boundaries = False
    default_model_name = 'mtcnn-o'

    def postprocess_result(self, identifiers, this_stage_result, batch_meta, previous_stage_result, *args, **kwargs):
        batch_predictions = calibrate_predictions(
            previous_stage_result, this_stage_result, 0.7, self.model_info['outputs']
        )
        batch_predictions[0], _ = nms(batch_predictions[0], 0.7, 'Min')
        if self.store:
            self._predictions.extend(batch_predictions)
        return batch_predictions

    def dump_predictions(self):
        if not hasattr(self, 'prediction_file'):
            prediction_file = Path(self.model_info.get('predictions', 'onet_predictions.pickle'))
            self.prediction_file = prediction_file
        with self.prediction_file.open('wb') as out_file:
            pickle.dump(self._predictions, out_file)


class CaffeModelMixin:
    def _infer(self, input_blobs, batch_meta, *args, **kwargs):
        for meta in batch_meta:
            meta['input_shape'] = []
        results = []
        for feed_dict in input_blobs:
            for layer_name, data in feed_dict.items():
                if data.shape != self.inputs[layer_name]:
                    self.net.blobs[layer_name].reshape(*data.shape)
            for meta in batch_meta:
                meta['input_shape'].append(self.inputs)
            results.append(self.net.forward(**feed_dict))
        return results

    @property
    def inputs(self):
        inputs_map = {}
        for input_blob in self.net.inputs:
            inputs_map[input_blob] = self.net.blobs[input_blob].data.shape
        return inputs_map

    def input_shape(self, input_name):
        return self.inputs[input_name]

    def release(self):
        del self.net

    def fit_to_input(self, data, layer_name, layout, precision, tmpl=None):
        data_shape = np.shape(data)
        layer_shape = self.inputs[layer_name]
        if len(data_shape) == 5 and len(layer_shape) == 4:
            data = data[0]
            data_shape = np.shape(data)
        data = np.transpose(data, layout) if len(data_shape) == 4 else np.array(data)
        if precision:
            data = data.astype(precision)
        return data

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


class DLSDKModelMixin:
    def _infer(self, input_blobs, batch_meta):
        for meta in batch_meta:
            meta['input_shape'] = []
        results = []
        for feed_dict in input_blobs:
            input_shapes = {layer_name: data.shape for layer_name, data in feed_dict.items()}
            if not self.is_dynamic:
                self.reshape_net(input_shapes)
            results.append(self.exec_network.infer(feed_dict))
            for meta in batch_meta:
                meta['input_shape'].append(input_shapes)
        return results

    @property
    def inputs(self):
        if self.exec_network:
            has_info = hasattr(self.exec_network, 'input_info')
            if not has_info:
                return self.exec_network.inputs
            inputs = OrderedDict()
            for name, data in self.exec_network.input_info.items():
                if name in self.partial_shapes:
                    inputs[name] = self.partial_shapes[name]
                else:
                    inputs[name] = data.input_data
            return inputs
        has_info = hasattr(self.network, 'input_info')
        if not has_info:
            return self.network.inputs
        inputs = OrderedDict()
        for name, data in self.network.input_info.items():
            if name in self.partial_shapes:
                inputs[name] = self.partial_shapes[name]
            else:
                inputs[name] = data.input_data
        return inputs

    def input_shape(self, input_name):
        return self.inputs[input_name]

    def release(self):
        self.input_feeder.release()
        del self.network
        del self.exec_network
        self.launcher.release()

    def fit_to_input(self, data, layer_name, layout, precision, template=None):
        layer_shape = (
            tuple(self.inputs[layer_name].shape)
            if layer_name not in self.dynamic_inputs else self.partial_shapes[layer_name])
        data_shape = np.shape(data)
        if len(layer_shape) == 4:
            if len(data_shape) == 5:
                data = data[0]
            data = np.transpose(data, layout)
        if precision:
            data = data.astype(precision)
        return data

    def prepare_model(self):
        model, weights = self.auto_model_search(self.model_info)
        return model, weights

    def auto_model_search(self, network_info):
        model = Path(network_info.get('model', ''))
        weights = network_info.get('weights')
        if model.is_dir():
            models_list = list(Path(model).glob('{}.xml'.format(self.default_model_name)))
            if not models_list:
                models_list = list(Path(model).glob('*.xml'))
            if not models_list:
                raise ConfigError('Suitable model description is not detected')
            if len(models_list) != 1:
                raise ConfigError('Several suitable models found, please specify required model')
            model = models_list[0]
        if weights is None or Path(weights).is_dir():
            weights_dir = weights or model.parent
            weights = Path(weights_dir) / model.name.replace('xml', 'bin')
            if not weights.exists():
                weights_list = list(weights_dir.glob('*.bin'))
                if not weights_list:
                    raise ConfigError('Suitable weights is not detected')
                if len(weights_list) != 1:
                    raise ConfigError('Several suitable weights found, please specify required explicitly')
                weights = weights_list[0]
        weights = get_path(weights)
        accepted_suffixes = ['.blob', '.xml']
        if model.suffix not in accepted_suffixes:
            raise ConfigError('Models with following suffixes are allowed: {}'.format(accepted_suffixes))
        print_info('{} - Found model: {}'.format(self.default_model_name, model))
        accepted_weights_suffixes = ['.bin']
        if weights.suffix not in accepted_weights_suffixes:
            raise ConfigError('Weights with following suffixes are allowed: {}'.format(accepted_weights_suffixes))
        print_info('{} - Found weights: {}'.format(self.default_model_name, weights))
        return model, weights

    def load_network(self, network, launcher, model_prefix):
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
        self.update_input_output_info(model_prefix)
        self.input_feeder = InputFeeder(
            self.model_info.get('inputs', []), self.inputs, self.input_shape, self.fit_to_input)

    def reshape_net(self, shape):
        if self.is_dynamic:
            return
        if hasattr(self, 'exec_network') and self.exec_network is not None:
            del self.exec_network
        self.network.reshape(shape)
        self.dynamic_inputs, self.partial_shapes = self.launcher.get_dynamic_inputs(self.network)
        if not self.is_dynamic and self.dynamic_inputs:
            return
        self.exec_network = self.launcher.ie_core.load_network(self.network, self.launcher.device)

    def load_model(self, network_info, launcher, model_prefix=None, log=False):
        self.network = launcher.read_network(str(network_info['model']), str(network_info['weights']))
        self.load_network(self.network, launcher, model_prefix)
        if log:
            self.print_input_output_info()

    def print_input_output_info(self):
        print_info('{} - Input info:'.format(self.default_model_name))
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
        print_info('{} - Output info'.format(self.default_model_name))
        for name, output_info in network_outputs.items():
            print_info('\tLayer name: {}'.format(name))
            print_info('\tprecision: {}'.format(output_info.precision))
            print_info('\tshape: {}\n'.format(
                output_info.shape if name not in self.partial_shapes else self.partial_shapes[name]))

    def update_input_output_info(self, model_prefix):
        def generate_name(prefix, with_prefix, layer_name):
            return prefix + layer_name if with_prefix else layer_name.split(prefix)[-1]

        if model_prefix is None:
            return
        config_inputs = self.model_info.get('inputs', [])
        network_with_prefix = next(iter(self.inputs)).startswith(model_prefix)
        if config_inputs:
            config_with_prefix = config_inputs[0]['name'].startswith(model_prefix)
            if config_with_prefix == network_with_prefix:
                return
            for c_input in config_inputs:
                c_input['name'] = generate_name(model_prefix, network_with_prefix, c_input['name'])
            self.model_info['inputs'] = config_inputs
        config_outputs = self.model_info['outputs']
        for key, value in config_outputs.items():
            config_with_prefix = value.startswith(model_prefix)
            if config_with_prefix != network_with_prefix:
                config_outputs[key] = generate_name(model_prefix, network_with_prefix, value)
        self.model_info['outputs'] = config_outputs


class OVModelMixin(BaseOpenVINOModel):
    def _infer(self, input_blobs, batch_meta):
        for meta in batch_meta:
            meta['input_shape'] = []
        results = []
        raw_results = []
        for feed_dict in input_blobs:
            input_shapes = {layer_name: data.shape for layer_name, data in feed_dict.items()}
            if not self.is_dynamic:
                self.reshape_net(input_shapes)
            result, raw_result = self.infer(feed_dict, raw_results=True)
            results.append(result)
            raw_results.append(raw_result)
            for meta in batch_meta:
                meta['input_shape'].append(input_shapes)
        return results, raw_results

    def predict(self, identifiers, input_data):
        raise NotImplementedError

    def input_shape(self, input_name):
        return parse_partial_shape(self.inputs[input_name].get_partial_shape())

    def release(self):
        self.input_feeder.release()
        del self.network
        del self.exec_network
        self.launcher.release()

    def fit_to_input(self, data, layer_name, layout, precision, template=None):
        layer_shape = (
            tuple(self.inputs[layer_name].shape)
            if layer_name not in self.dynamic_inputs else self.partial_shapes[layer_name])
        data_shape = np.shape(data)
        if len(layer_shape) == 4:
            if len(data_shape) == 5:
                data = data[0]
            data = np.transpose(data, layout)
        if precision:
            data = data.astype(precision)
        return data

    def prepare_model(self):
        model, weights = self.auto_model_search(self.model_info)
        return model, weights

    def auto_model_search(self, network_info):
        model = Path(network_info.get('model', ''))
        weights = network_info.get('weights')
        if model.is_dir():
            models_list = list(Path(model).glob('{}.xml'.format(self.default_model_name)))
            if not models_list:
                models_list = list(Path(model).glob('*.xml'))
            if not models_list:
                raise ConfigError('Suitable model description is not detected')
            if len(models_list) != 1:
                raise ConfigError('Several suitable models found, please specify required model')
            model = models_list[0]
        if weights is None or Path(weights).is_dir():
            weights_dir = weights or model.parent
            weights = Path(weights_dir) / model.name.replace('xml', 'bin')
            if not weights.exists():
                weights_list = list(weights_dir.glob('*.bin'))
                if not weights_list:
                    raise ConfigError('Suitable weights is not detected')
                if len(weights_list) != 1:
                    raise ConfigError('Several suitable weights found, please specify required explicitly')
                weights = weights_list[0]
        weights = get_path(weights)
        accepted_suffixes = ['.blob', '.xml']
        if model.suffix not in accepted_suffixes:
            raise ConfigError('Models with following suffixes are allowed: {}'.format(accepted_suffixes))
        print_info('{} - Found model: {}'.format(self.default_model_name, model))
        accepted_weights_suffixes = ['.bin']
        if weights.suffix not in accepted_weights_suffixes:
            raise ConfigError('Weights with following suffixes are allowed: {}'.format(accepted_weights_suffixes))
        print_info('{} - Found weights: {}'.format(self.default_model_name, weights))
        return model, weights

    def load_network(self, network, launcher, model_prefix):
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
        self.update_input_output_info(model_prefix)
        self.input_feeder = InputFeeder(
            self.model_info.get('inputs', []), self.inputs, self.input_shape, self.fit_to_input)
        self.infer_request = None

    def reshape_net(self, shape):
        self._reshape_input(shape)

    def load_model(self, network_info, launcher, model_prefix=None, log=False):
        self.network = launcher.read_network(str(network_info['model']), str(network_info['weights']))
        self.load_network(self.network, launcher, model_prefix)
        if log:
            self.print_input_output_info()
        self.infer_request = None

    def update_input_output_info(self, model_prefix):
        def generate_name(prefix, with_prefix, layer_name):
            return prefix + layer_name if with_prefix else layer_name.split(prefix)[-1]

        if model_prefix is None:
            return
        config_inputs = self.model_info.get('inputs', [])
        network_with_prefix = next(iter(self.inputs)).startswith(model_prefix)
        if config_inputs:
            config_with_prefix = config_inputs[0]['name'].startswith(model_prefix)
            if config_with_prefix == network_with_prefix:
                return
            for c_input in config_inputs:
                c_input['name'] = generate_name(model_prefix, network_with_prefix, c_input['name'])
            self.model_info['inputs'] = config_inputs
        config_outputs = self.model_info['outputs']
        for key, value in config_outputs.items():
            output = postprocess_output_name(
                value, self.outputs, additional_mapping=self.additional_output_mapping, raise_error=False
            )
            if output not in self.outputs:
                output =postprocess_output_name(
                    generate_name(model_prefix, network_with_prefix, value),
                    value, self.outputs, additional_mapping=self.additional_output_mapping, raise_error=False
                )
            config_outputs[key] = output
        self.model_info['outputs'] = config_outputs


class CaffeProposalStage(CaffeModelMixin, ProposalBaseStage):
    def __init__(self, model_info, model_specific_preprocessor, common_preprocessor, launcher, *args, **kwargs):
        super().__init__(model_info, model_specific_preprocessor, common_preprocessor)
        self.net = launcher.create_network(self.model_info['model'], self.model_info['weights'])
        self.input_feeder = InputFeeder(model_info.get('inputs', []), self.inputs, self.input_shape, self.fit_to_input)
        pnet_outs = model_info['outputs']
        pnet_adapter_config = launcher.config.get('adapter', {'type': 'mtcnn_p', **pnet_outs})
        pnet_adapter_config.update({'regions_format': 'hw'})
        self.adapter = create_adapter(pnet_adapter_config)


class CaffeRefineStage(CaffeModelMixin, RefineBaseStage):
    def __init__(self, model_info, model_specific_preprocessor, common_preprocessor, launcher, *args, **kwargs):
        super().__init__(model_info, model_specific_preprocessor, common_preprocessor)
        self.net = launcher.create_network(self.model_info['model'], self.model_info['weights'])
        self.input_feeder = InputFeeder(model_info.get('inputs', []), self.inputs, self.input_shape, self.fit_to_input)


class CaffeOutputStage(CaffeModelMixin, OutputBaseStage):
    def __init__(self, model_info, model_specific_preprocessor, common_preprocessor, launcher):
        super().__init__(model_info, model_specific_preprocessor, common_preprocessor)
        self.net = launcher.create_network(self.model_info['model'], self.model_info['weights'])
        self.input_feeder = InputFeeder(model_info.get('inputs', []), self.inputs, self.input_shape, self.fit_to_input)


class OpenVINOProposalStage(ProposalBaseStage, OVModelMixin):
    def __init__(
        self, model_info, model_specific_preprocessor, common_preprocessor, launcher, delayed_model_loading=False
    ):
        super().__init__(model_info, model_specific_preprocessor, common_preprocessor)
        self.adapter = None
        self.is_dynamic = False
        if not delayed_model_loading:
            model_xml, model_bin = self.prepare_model()
            self.load_model({'model': model_xml, 'weights': model_bin}, launcher, 'pnet_', log=True)
            pnet_outs = model_info['outputs']
            pnet_adapter_config = launcher.config.get('adapter', {'type': 'mtcnn_p', **pnet_outs})
            # pnet_adapter_config.update({'regions_format': 'hw'})
            self.adapter = create_adapter(pnet_adapter_config)

    def load_network(self, network, launcher, model_prefix):
        self.network = network
        self.dynamic_inputs, self.partial_shapes = launcher.get_dynamic_inputs(self.network)
        if self.dynamic_inputs and launcher.dynamic_shapes_policy in ['dynamic', 'default']:
            try:
                self.exec_network = launcher.ie_core.compile_modelk(self.network, launcher.device)
                self.is_dynamic = True
            except RuntimeError as e:
                if launcher.dynamic_shapes_policy == 'dynamic':
                    raise e
                self.is_dynamic = False
                self.exec_network = None
        if not self.dynamic_inputs:
            self.exec_network = launcher.ie_core.compile_model(self.network, launcher.device)
        self.update_input_output_info(model_prefix)
        self.input_feeder = InputFeeder(
            self.model_info.get('inputs', []), self.inputs, self.input_shape, self.fit_to_input)
        pnet_outs = self.model_info['outputs']
        pnet_adapter_config = launcher.config.get('adapter', {'type': 'mtcnn_p', **pnet_outs})
        self.adapter = create_adapter(pnet_adapter_config)

    def load_model(self, network_info, launcher, model_prefix=None, log=False):
        self.network = launcher.read_network(str(network_info['model']), str(network_info['weights']))
        self.load_network(self.network, launcher, model_prefix)
        self.launcher = launcher
        if log:
            self.print_input_output_info()

    def predict(self, input_blobs, batch_meta, output_callback=None):
        outputs, raw_outputs = self._infer(input_blobs, batch_meta)
        if output_callback:
            for out in raw_outputs:
                output_callback(out)
        return outputs, raw_outputs


class DLSDKProposalStage(DLSDKModelMixin, ProposalBaseStage):
    def __init__(
        self, model_info, model_specific_preprocessor, common_preprocessor, launcher, delayed_model_loading=False
    ):
        super().__init__(model_info, model_specific_preprocessor, common_preprocessor)
        self.adapter = None
        self.is_dynamic = False
        if not delayed_model_loading:
            model_xml, model_bin = self.prepare_model()
            self.load_model({'model': model_xml, 'weights': model_bin}, launcher, 'pnet_', log=True)
            pnet_outs = model_info['outputs']
            pnet_adapter_config = launcher.config.get('adapter', {'type': 'mtcnn_p', **pnet_outs})
            # pnet_adapter_config.update({'regions_format': 'hw'})
            self.adapter = create_adapter(pnet_adapter_config)

    def load_network(self, network, launcher, model_prefix):
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
        self.input_feeder = InputFeeder(
            self.model_info.get('inputs', []), self.inputs, self.input_shape, self.fit_to_input)
        pnet_outs = self.model_info['outputs']
        pnet_adapter_config = launcher.config.get('adapter', {'type': 'mtcnn_p', **pnet_outs})
        self.adapter = create_adapter(pnet_adapter_config)

    def load_model(self, network_info, launcher, model_prefix=None, log=False):
        self.network = launcher.read_network(str(network_info['model']), str(network_info['weights']))
        self.load_network(self.network, launcher, model_prefix)
        self.launcher = launcher
        if log:
            self.print_input_output_info()

    def predict(self, input_blobs, batch_meta, output_callback=None):
        raw_outputs = self._infer(input_blobs, batch_meta)
        if output_callback:
            for out in raw_outputs:
                output_callback(out)
        return raw_outputs


class OpenVINORefineStage(RefineBaseStage, OVModelMixin):
    def __init__(
        self, model_info, model_specific_preprocessor, common_preprocessor, launcher, delayed_model_loading=False
    ):
        self.default_model_suffix = 'rnet'
        super().__init__(model_info, model_specific_preprocessor, common_preprocessor)
        self.is_dynamic = False
        self.launcher = launcher
        if not delayed_model_loading:
            model_xml, model_bin = self.prepare_model()
            self.load_model({'model': model_xml, 'weights': model_bin}, launcher, 'rnet_', log=True)

    def predict(self, input_blobs, batch_meta, output_callback=None):
        raw_outputs = self._infer(input_blobs, batch_meta)
        if output_callback:
            batch_size = np.shape(next(iter(input_blobs[0].values())))[0]
            output_callback(transform_for_callback(batch_size, raw_outputs[1]))
        return raw_outputs


class DLSDKRefineStage(DLSDKModelMixin, RefineBaseStage):
    def __init__(
        self, model_info, model_specific_preprocessor, common_preprocessor, launcher, delayed_model_loading=False
    ):
        super().__init__(model_info, model_specific_preprocessor, common_preprocessor)
        self.is_dynamic = False
        self.launcher = launcher
        if not delayed_model_loading:
            model_xml, model_bin = self.prepare_model()
            self.load_model({'model': model_xml, 'weights': model_bin}, launcher, 'rnet_', log=True)

    def predict(self, input_blobs, batch_meta, output_callback=None):
        raw_outputs = self._infer(input_blobs, batch_meta)
        if output_callback:
            batch_size = np.shape(next(iter(input_blobs[0].values())))[0]
            output_callback(transform_for_callback(batch_size, raw_outputs))
        return raw_outputs


class DLSDKOutputStage(DLSDKModelMixin, OutputBaseStage):
    def __init__(
        self, model_info, model_specific_preprocessor, common_preprocessor, launcher, delayed_model_loading=False
    ):
        super().__init__(model_info, model_specific_preprocessor, common_preprocessor)
        self.is_dynamic = False
        self.launcher = launcher
        if not delayed_model_loading:
            model_xml, model_bin = self.prepare_model()
            self.load_model({'model': model_xml, 'weights': model_bin}, launcher, 'onet_', log=True)

    def predict(self, input_blobs, batch_meta, output_callback=None):
        raw_outputs = self._infer(input_blobs, batch_meta)
        return raw_outputs


class OpenVINOOutputStage(OutputBaseStage, OVModelMixin):
    def __init__(
        self, model_info, model_specific_preprocessor, common_preprocessor, launcher, delayed_model_loading=False
    ):
        self.default_model_suffix = 'onet'
        super().__init__(model_info, model_specific_preprocessor, common_preprocessor)
        self.is_dynamic = False
        self.launcher = launcher
        if not delayed_model_loading:
            model_xml, model_bin = self.prepare_model()
            self.load_model({'model': model_xml, 'weights': model_bin}, launcher, 'onet_', log=True)

    def predict(self, input_blobs, batch_meta, output_callback=None):
        raw_outputs = self._infer(input_blobs, batch_meta)
        return raw_outputs
