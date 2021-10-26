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

import copy
from functools import partial
from collections import OrderedDict
import pickle # nosec - disable B403:import-pickle check
from pathlib import Path
import numpy as np

from .mtcnn_evaluator_utils import calibrate_predictions, nms, cut_roi
from .base_custom_evaluator import BaseCustomEvaluator
from ..quantization_model_evaluator import create_dataset_attributes
from ...adapters import create_adapter
from ...launcher import InputFeeder
from ...preprocessor import PreprocessingExecutor
from ...utils import extract_image_representations, read_pickle, contains_any, get_path
from ...config import ConfigError
from ...logging import print_info


def build_stages(models_info, preprocessors_config, launcher, model_args, delayed_model_loading=False):
    required_stages = ['pnet']
    stages_mapping = OrderedDict([
        ('pnet', {'caffe': CaffeProposalStage, 'dlsdk': DLSDKProposalStage, 'dummy': DummyProposalStage}),
        ('rnet', {'caffe': CaffeRefineStage, 'dlsdk': DLSDKRefineStage}),
        ('onet', {'caffe': CaffeOutputStage, 'dlsdk': DLSDKOutputStage})
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

    def _infer(self, input_blobs, batch_meta):
        raise NotImplementedError

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
        results = self._predictions[self._index:self._index+batch_size]
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

    def _infer(self, input_blobs, batch_meta):
        raise NotImplementedError

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

    def _infer(self, input_blobs, batch_meta):
        raise NotImplementedError

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

    def fit_to_input(self, data, layer_name, layout, precision):
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

    def prepare_model(self, launcher):
        launcher_specific_entries = [
            'model', 'weights', 'caffe_model', 'caffe_weights', 'tf_model', 'inputs', 'outputs', '_model_optimizer'
        ]

        def update_mo_params(launcher_config, model_config):
            for entry in launcher_specific_entries:
                if entry not in launcher_config:
                    continue
                if entry in model_config:
                    continue
                model_config[entry] = launcher_config[entry]
            model_mo_flags, model_mo_params = model_config.get('mo_flags', []), model_config.get('mo_params', {})
            launcher_mo_flags, launcher_mo_params = launcher_config.get('mo_flags', []), launcher_config.get(
                'mo_params', {})
            for launcher_flag in launcher_mo_flags:
                if launcher_flag not in model_mo_flags:
                    model_mo_flags.append(launcher_flag)

            for launcher_mo_key, launcher_mo_value in launcher_mo_params.items():
                if launcher_mo_key not in model_mo_params:
                    model_mo_params[launcher_mo_key] = launcher_mo_value

            model_config['mo_flags'] = model_mo_flags
            model_config['mo_params'] = model_mo_params

        update_mo_params(launcher.config, self.model_info)
        if 'caffe_model' in self.model_info:
            model, weights = launcher.convert_model(self.model_info)
        else:
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


class DLSDKProposalStage(DLSDKModelMixin, ProposalBaseStage):
    def __init__(
            self, model_info, model_specific_preprocessor, common_preprocessor, launcher, delayed_model_loading=False
    ):
        super().__init__(model_info, model_specific_preprocessor, common_preprocessor)
        self.adapter = None
        self.is_dynamic = False
        if not delayed_model_loading:
            model_xml, model_bin = self.prepare_model(launcher)
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
        raw_outputs = self._infer(input_blobs, batch_meta)
        if output_callback:
            for out in raw_outputs:
                output_callback(out)
        return raw_outputs


class DLSDKRefineStage(DLSDKModelMixin, RefineBaseStage):
    def __init__(
            self, model_info, model_specific_preprocessor, common_preprocessor, launcher, delayed_model_loading=False
    ):
        super().__init__(model_info, model_specific_preprocessor, common_preprocessor)
        self.is_dynamic = False
        self.launcher = launcher
        if not delayed_model_loading:
            model_xml, model_bin = self.prepare_model(launcher)
            self.load_model({'model': model_xml, 'weights': model_bin}, launcher, 'rnet_', log=True)

    def predict(self, input_blobs, batch_meta, output_callback=None):
        raw_outputs = self._infer(input_blobs, batch_meta)
        if output_callback:
            batch_size = np.shape(next(iter(input_blobs[0].values())))[0]
            output_callback(self.transform_for_callback(batch_size, raw_outputs))
        return raw_outputs

    @staticmethod
    def transform_for_callback(batch_size, raw_outputs):
        output_per_box = []
        fq_weights = []
        for i in range(batch_size):
            box_outs = OrderedDict()
            for layer_name, data in raw_outputs[0].items():
                if layer_name in fq_weights:
                    continue
                if layer_name.endswith('fq_weights_1'):
                    fq_weights.append(layer_name)
                    box_outs[layer_name] = data
                elif data.shape[0] <= i:
                    box_outs[layer_name] = data
                else:
                    box_outs[layer_name] = np.expand_dims(data[i], axis=0)
            output_per_box.append(box_outs)
        return output_per_box


class DLSDKOutputStage(DLSDKModelMixin, OutputBaseStage):
    def __init__(
            self, model_info, model_specific_preprocessor, common_preprocessor, launcher, delayed_model_loading=False
    ):
        super().__init__(model_info, model_specific_preprocessor, common_preprocessor)
        self.is_dynamic = False
        self.launcher = launcher
        if not delayed_model_loading:
            model_xml, model_bin = self.prepare_model(launcher)
            self.load_model({'model': model_xml, 'weights': model_bin}, launcher, 'onet_', log=True)

    def predict(self, input_blobs, batch_meta, output_callback=None):
        raw_outputs = self._infer(input_blobs, batch_meta)
        return raw_outputs

    @staticmethod
    def transform_for_callback(batch_size, raw_outputs):
        output_per_box = []
        fq_weights = []
        for i in range(batch_size):
            box_outs = OrderedDict()
            for layer_name, data in raw_outputs[0].items():
                if layer_name in fq_weights:
                    continue
                if layer_name.endswith('fq_weights_1'):
                    fq_weights.append(layer_name)
                    box_outs[layer_name] = data
                elif data.shape[0] <= i:
                    box_outs[layer_name] = data
                else:
                    box_outs[layer_name] = np.expand_dims(data[i], axis=0)
            output_per_box.append(box_outs)
        return output_per_box


class MTCNNEvaluator(BaseCustomEvaluator):
    def __init__(self, dataset_config, launcher, stages, orig_config):
        super().__init__(dataset_config, launcher, orig_config)
        self.stages = stages
        stage = next(iter(self.stages.values()))
        if hasattr(stage, 'adapter') and stage.adapter is not None:
            self.adapter_type = stage.adapter.__provider__

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, _ = cls.get_dataset_and_launcher_info(config)
        models_info = config['network_info']
        stages = build_stages(models_info, [], launcher, config.get('_models'), delayed_model_loading)
        return cls(dataset_config, launcher, stages, orig_config)

    def _process(self, output_callback, calculate_metrics, progress_reporter, metric_config, csv_file):
        def no_detections(batch_pred):
            return batch_pred[0].size == 0

        for batch_id, (batch_input_ids, batch_annotation, batch_inputs, batch_identifiers) in enumerate(self.dataset):
            batch_prediction = []
            batch_raw_prediction = []
            intermediate_callback = None
            if output_callback:
                intermediate_callback = partial(output_callback, metrics_result=None,
                                                element_identifiers=batch_identifiers, dataset_indices=batch_input_ids)
            batch_size = 1
            for stage in self.stages.values():
                previous_stage_predictions = batch_prediction
                filled_inputs, batch_meta = stage.preprocess_data(
                    copy.deepcopy(batch_inputs), batch_annotation, previous_stage_predictions
                )
                batch_raw_prediction = stage.predict(filled_inputs, batch_meta, intermediate_callback)
                batch_size = np.shape(next(iter(filled_inputs[0].values())))[0]
                batch_prediction = stage.postprocess_result(
                    batch_identifiers, batch_raw_prediction, batch_meta, previous_stage_predictions
                )
                if no_detections(batch_prediction):
                    break
            batch_annotation, batch_prediction = self.postprocessor.process_batch(batch_annotation, batch_prediction)
            metrics_result = self._get_metrics_result(batch_input_ids, batch_annotation, batch_prediction,
                                                      calculate_metrics)
            if output_callback:
                output_callback(list(self.stages.values())[-1].transform_for_callback(batch_size, batch_raw_prediction),
                                metrics_result=metrics_result, element_identifiers=batch_identifiers,
                                dataset_indices=batch_input_ids)
            self._update_progress(progress_reporter, metric_config, batch_id, len(batch_prediction), csv_file)

    def _release_model(self):
        for _, stage in self.stages.items():
            stage.release()

    def reset(self):
        super().reset()
        for _, stage in self.stages.items():
            stage.reset()

    def load_network(self, network=None):
        if network is None:
            for stage_name, stage in self.stages.items():
                stage.load_network(network, self.launcher, stage_name + '_')
        else:
            for net_dict in network:
                stage_name = net_dict['name']
                network_ = net_dict['model']
                self.stages[stage_name].load_network(network_, self.launcher, stage_name+'_')

    def load_network_from_ir(self, models_list):
        for models_dict in models_list:
            stage_name = models_dict['name']
            self.stages[stage_name].load_model(models_dict, self.launcher, stage_name+'_')

    def get_network(self):
        return [{'name': stage_name, 'model': stage.network} for stage_name, stage in self.stages.items()]

    def select_dataset(self, dataset_tag):
        if self.dataset is not None and isinstance(self.dataset_config, list):
            return
        dataset_attributes = create_dataset_attributes(self.dataset_config, dataset_tag)
        self.dataset, self.metric_executor, preprocessor, self.postprocessor = dataset_attributes
        for _, stage in self.stages.items():
            stage.update_preprocessing(preprocessor)
