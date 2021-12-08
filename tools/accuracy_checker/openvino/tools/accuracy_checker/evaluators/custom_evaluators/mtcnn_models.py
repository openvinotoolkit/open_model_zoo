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

import pickle # nosec - disable B403:import-pickle check
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ...preprocessor import PreprocessingExecutor
from ...config import ConfigError
from ...utils import (
    contains_any, extract_image_representations, read_pickle, parse_partial_shape, generate_layer_name, contains_all
)
from .mtcnn_evaluator_utils import cut_roi, calibrate_predictions, nms, transform_for_callback
from ...launcher import InputFeeder
from .base_models import BaseOpenVINOModel, BaseDLSDKModel, BaseCaffeModel, BaseCascadeModel
from ...adapters import create_adapter


def create_pnet(model_config, launcher, launcher_model_mapping, common_preprocessor, delayed_model_loading=False):
    framework = launcher.config['framework']
    if 'predictions' in model_config and not model_config.get('store_predictions', False):
        framework = 'dummy'
    model_class = launcher_model_mapping.get(framework)
    if not model_class:
        raise ValueError('model for framework {} is not supported'.format(framework))
    stage_preprocess = model_config.get('preprocessing', [])
    model_specific_preprocessor = PreprocessingExecutor(stage_preprocess)
    return model_class(model_config, model_specific_preprocessor, common_preprocessor, launcher, delayed_model_loading)


def create_net(model_config, launcher, launcher_model_mapping, common_preprocessor, delayed_model_loading=False):
    framework = launcher.config['framework']
    model_class = launcher_model_mapping.get(framework)
    if not model_class:
        raise ValueError('model for framework {} is not supported'.format(framework))
    stage_preprocess = model_config.get('preprocessing', [])
    model_specific_preprocessor = PreprocessingExecutor(stage_preprocess)
    return model_class(model_config, model_specific_preprocessor, common_preprocessor, launcher, delayed_model_loading)


class MTCNNCascadeModel(BaseCascadeModel):
    def __init__(self, network_info, launcher, models_args, delayed_model_loading=False):
        super().__init__(network_info, launcher)
        required_stages = ['pnet']
        stages = ['pnet', 'rnet', 'onet']
        network_info = self.fill_part_with_model(network_info, stages, models_args, delayed_model_loading)
        if not contains_all(network_info, required_stages) and not delayed_model_loading:
            raise ConfigError('network_info should contain pnet field')
        self._pnet_mapping = {
            'caffe': CaffeProposalStage,
            'dlsdk': DLSDKProposalStage,
            'dummy': DummyProposalStage,
            'openvino': OpenVINOProposalStage
        }
        self._rnet_mapping = {
            'caffe': CaffeRefineStage,
            'dlsdk': DLSDKRefineStage,
            'openvino': OpenVINORefineStage
        }
        self._onet_mapping = {
            'caffe': CaffeOutputStage,
            'dlsdk': DLSDKOutputStage,
            'openvino': OpenVINOOutputStage
        }
        common_preprocessor = PreprocessingExecutor([])
        self.pnet = create_pnet(network_info['pnet'], launcher, self._pnet_mapping, common_preprocessor,
                                delayed_model_loading)
        self.rnet = create_net(network_info['rnet'], launcher, self._rnet_mapping, common_preprocessor,
                                delayed_model_loading)
        self.onet = create_net(network_info['onet'], launcher, self._onet_mapping, common_preprocessor,
                               delayed_model_loading)
        self._part_by_name = {'pnet': self.pnet, 'rnet': self.rnet, 'onet': self.onet}

    def predict(self, identifiers, input_data, encoder_callback=None):
        pass

    @property
    def adapter(self):
        return self.pnet.adapter

    @property
    def stages(self):
        return self._part_by_name

    def reset(self):
        for stage in self._part_by_name.values():
            stage.reset()

    @staticmethod
    def fill_part_with_model(network_info, parts, models_args, delayed_model_loading):
        if models_args and not delayed_model_loading:
            for idx, part in enumerate(parts):
                part_info = network_info.get(part, {})
                if not contains_any(part_info, ['model', 'caffe_model']) and models_args:
                    part_info['model'] = models_args[idx if len(models_args) > idx else 0]
                network_info.update({part: part_info})
        return network_info


class BaseStage:
    def predict(self, input_blobs, batch_meta, output_callback=None):
        return self._infer(input_blobs, batch_meta)

    def preprocess_data(self, batch_input, batch_annotation, previous_stage_prediction, *args, **kwargs):
        batch_input = self.model_specific_preprocessor.process(batch_input, batch_annotation)
        batch_input = self.common_preprocessor.process(batch_input, batch_annotation)
        _, batch_meta = extract_image_representations(batch_input)
        batch_input = self.update_batch_input(batch_input, previous_stage_prediction)
        filled_inputs = self.input_feeder.fill_inputs(batch_input) if self.input_feeder else batch_input
        return filled_inputs, batch_meta

    def update_batch_input(self, batch_input, previous_stage_prediction): # pylint:disable=R0201
        return batch_input

    def dump_predictions(self, pickle_name=None):
        if not hasattr(self, 'prediction_file'):
            prediction_file = Path(self.network_info.get('predictions', '{}.pickle'.format(pickle_name)))
            self.prediction_file = prediction_file
        with self.prediction_file.open('wb') as out_file:
            pickle.dump(self._predictions, out_file)

    def reset(self):
        self._predictions = []

    def update_preprocessing(self, preprocessor):
        self.common_preprocessor = preprocessor


class ProposalBaseStage(BaseStage):
    def postprocess_result(self, identifiers, this_stage_result, batch_meta, *args, **kwargs):
        result = self.adapter.process(this_stage_result, identifiers, batch_meta) if self.adapter else this_stage_result
        if self.store:
            self._predictions.extend(result)
        return result

    def dump_predictions(self):
        super().dump_predictions('pnet_predictions')


class DummyProposalStage(ProposalBaseStage):
    def __init__(self, model_info, model_specific_preprocessor, common_preprocessor, *args, **kwargs):
        self.default_model_name = 'mtcnn-p'
        self.default_model_suffix = 'pnet'
        self.network_info = model_info
        self.model_specific_preprocessor = model_specific_preprocessor
        self.common_preprocessor = common_preprocessor
        self.store = model_info.get('store_predictions', False)
        self.adapter = None
        self.input_feeder = None
        self._predictions = []
        self._index = 0
        if 'predictions' not in self.network_info:
            raise ConfigError('predictions_file is not found')
        self._predictions = read_pickle(self.network_info['predictions'])
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

    def release(self):
        pass


class RefineBaseStage(BaseStage):
    def update_batch_input(self, batch_input, previous_stage_prediction):
        return [
            cut_roi(input_image, prediction, self.input_size, include_bound=self.include_boundaries)
            for input_image, prediction in zip(batch_input, previous_stage_prediction)
        ]

    def postprocess_result(self, identifiers, this_stage_result, batch_meta, previous_stage_result, *args, **kwargs):
        result = calibrate_predictions(
            previous_stage_result, this_stage_result, 0.7, self.network_info['outputs'], 'Union'
        )
        if self.store:
            self._predictions.extend(result)
        return result

    def dump_predictions(self):
        super().dump_predictions('rnet_predictions')


class OutputBaseStage(BaseStage):
    def update_batch_input(self, batch_input, previous_stage_prediction):
        return [
            cut_roi(input_image, prediction, self.input_size, include_bound=self.include_boundaries)
            for input_image, prediction in zip(batch_input, previous_stage_prediction)
        ]

    def postprocess_result(self, identifiers, this_stage_result, batch_meta, previous_stage_result, *args, **kwargs):
        batch_predictions = calibrate_predictions(
            previous_stage_result, this_stage_result, 0.7, self.network_info['outputs']
        )
        batch_predictions[0], _ = nms(batch_predictions[0], 0.7, 'Min')
        if self.store:
            self._predictions.extend(batch_predictions)
        return batch_predictions

    def dump_predictions(self):
        super().dump_predictions('onet_predictions')


class CaffeModelMixin(BaseCaffeModel):
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

    def predict(self, identifiers, input_data):
        raise NotImplementedError


class CaffeProposalStage(ProposalBaseStage, CaffeModelMixin):
    def __init__(self, model_info, model_specific_preprocessor, common_preprocessor, launcher, *args, **kwargs):
        super().__init__(model_info, launcher, 'pnet')
        self.model_specific_preprocessor = model_specific_preprocessor
        self.common_preprocessor = common_preprocessor
        self.store = model_info.get('store_predictions', False)
        self.adapter = None
        self._predictions = []
        self.default_model_name = 'mtcnn-p'
        self.net = launcher.create_network(self.network_info['model'], self.network_info['weights'])
        self.input_feeder = InputFeeder(model_info.get('inputs', []), self.inputs, self.input_shape, self.fit_to_input)
        pnet_outs = model_info['outputs']
        pnet_adapter_config = launcher.config.get('adapter', {'type': 'mtcnn_p', **pnet_outs})
        pnet_adapter_config.update({'regions_format': 'hw'})
        self.adapter = create_adapter(pnet_adapter_config)


class CaffeRefineStage(RefineBaseStage, CaffeModelMixin):
    def __init__(self, model_info, model_specific_preprocessor, common_preprocessor, launcher, *args, **kwargs):
        super().__init__(model_info, launcher, 'rnet')
        self.model_specific_preprocessor = model_specific_preprocessor
        self.common_preprocessor = common_preprocessor
        self.store = model_info.get('store_predictions', False)
        self._predictions = []
        self.input_size = 24
        self.include_boundaries = True
        self.default_model_name = 'mtcnn-r'
        self.net = launcher.create_network(self.network_info['model'], self.network_info['weights'])
        self.input_feeder = InputFeeder(model_info.get('inputs', []), self.inputs, self.input_shape, self.fit_to_input)


class CaffeOutputStage(OutputBaseStage, CaffeModelMixin):
    def __init__(self, model_info, model_specific_preprocessor, common_preprocessor, launcher):
        super().__init__(model_info, launcher, 'onet')
        self.model_specific_preprocessor = model_specific_preprocessor
        self.common_preprocessor = common_preprocessor
        self.store = model_info.get('store_predictions', False)
        self._predictions = []
        self.input_size = 48
        self.include_boundaries = False
        self.default_model_name = 'mtcnn-o'
        self.net = launcher.create_network(self.network_info['model'], self.network_info['weights'])
        self.input_feeder = InputFeeder(model_info.get('inputs', []), self.inputs, self.input_shape, self.fit_to_input)


class DLSDKModelMixin(BaseDLSDKModel):
    def __init__(
        self, model_info, model_specific_preprocessor, common_preprocessor, launcher, suffix=None,
        delayed_model_loading=False
    ):
        self.model_specific_preprocessor = model_specific_preprocessor
        self.common_preprocessor = common_preprocessor
        self.input_feeder = None
        self.store = model_info.get('store_predictions', False)
        self._predictions = []
        super().__init__(model_info, launcher, suffix, delayed_model_loading)

    def _infer(self, input_blobs, batch_meta):
        for meta in batch_meta:
            meta['input_shape'] = []
        results = []
        for feed_dict in input_blobs:
            input_shapes = {layer_name: data.shape for layer_name, data in feed_dict.items()}
            if not self.is_dynamic:
                self._reshape_input(input_shapes)
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
        super().release()

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

    def prepare_model(self, launcher, network_info):
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

        update_mo_params(launcher.config, network_info)
        if 'caffe_model' in network_info:
            model, weights = launcher.convert_model(network_info)
        else:
            model, weights = self.automatic_model_search(network_info)
        return model, weights

    def load_network(self, network, launcher):
        super().load_network(network, launcher)
        self.update_input_output_info()
        self.input_feeder = InputFeeder(
            self.network_info.get('inputs', []), self.inputs, self.input_shape, self.fit_to_input)

    def load_model(self, network_info, launcher, log=False):
        model_xml, model_bin = self.prepare_model(launcher, network_info)
        self.network = launcher.read_network(str(model_xml), str(model_bin))
        self.load_network(self.network, launcher)
        if log:
            self.print_input_output_info()

    def update_input_output_info(self):
        if self.default_model_suffix is None:
            return
        config_inputs = self.network_info.get('inputs', [])
        network_with_prefix = next(iter(self.inputs)).startswith(self.default_model_suffix+'_')
        if config_inputs:
            config_with_prefix = config_inputs[0]['name'].startswith(self.default_model_suffix+'_')
            if config_with_prefix == network_with_prefix:
                return
            for c_input in config_inputs:
                c_input['name'] = generate_layer_name(c_input['name'], self.default_model_suffix+'_',
                                                      network_with_prefix)
            self.network_info['inputs'] = config_inputs
        config_outputs = self.network_info['outputs']
        for key, value in config_outputs.items():
            config_with_prefix = value.startswith(self.default_model_suffix+'_')
            if config_with_prefix != network_with_prefix:
                config_outputs[key] = generate_layer_name(value, self.default_model_suffix+'_', network_with_prefix)
        self.network_info['outputs'] = config_outputs

    def predict(self, identifiers, input_data):
        raise NotImplementedError


class DLSDKProposalStage(DLSDKModelMixin, ProposalBaseStage):
    def __init__(
        self, model_info, model_specific_preprocessor, common_preprocessor, launcher, delayed_model_loading=False
    ):
        self.adapter = None
        self.default_model_name = 'mtcnn-p'
        super().__init__(model_info, model_specific_preprocessor, common_preprocessor, launcher, 'pnet',
                         delayed_model_loading)

    def load_network(self, network, launcher):
        super().load_network(network, launcher)
        pnet_outs = self.network_info['outputs']
        pnet_adapter_config = launcher.config.get('adapter', {'type': 'mtcnn_p', **pnet_outs})
        self.adapter = create_adapter(pnet_adapter_config)

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
        self.input_size = 24
        self.include_boundaries = True
        self.default_model_name = 'mtcnn-r'
        super().__init__(model_info, model_specific_preprocessor, common_preprocessor, launcher, 'rnet',
                         delayed_model_loading)

    def predict(self, input_blobs, batch_meta, output_callback=None):
        raw_outputs = self._infer(input_blobs, batch_meta)
        if output_callback:
            batch_size = np.shape(next(iter(input_blobs[0].values())))[0]
            output_callback(transform_for_callback(batch_size, raw_outputs))
        return raw_outputs


class DLSDKOutputStage(OutputBaseStage, DLSDKModelMixin):
    def __init__(
        self, model_info, model_specific_preprocessor, common_preprocessor, launcher, delayed_model_loading=False
    ):
        self.input_size = 48
        self.include_boundaries = False
        self.default_model_name = 'mtcnn-o'
        super().__init__(model_info, model_specific_preprocessor, common_preprocessor, launcher, 'onet',
                         delayed_model_loading)


class OVModelMixin(BaseOpenVINOModel):
    def __init__(
        self, model_info, model_specific_preprocessor, common_preprocessor, launcher, suffix=None,
        delayed_model_loading=False
    ):
        self.model_specific_preprocessor = model_specific_preprocessor
        self.common_preprocessor = common_preprocessor
        self.input_feeder = None
        self.store = model_info.get('store_predictions', False)
        self._predictions = []
        super().__init__(model_info, launcher, suffix, delayed_model_loading)

    def _infer(self, input_blobs, batch_meta):
        for meta in batch_meta:
            meta['input_shape'] = []
        results = []
        for feed_dict in input_blobs:
            input_shapes = {layer_name: data.shape for layer_name, data in feed_dict.items()}
            if not self.is_dynamic:
                self._reshape_input(input_shapes)
            results.append(self.infer(feed_dict))
            for meta in batch_meta:
                meta['input_shape'].append(input_shapes)
        return results

    def predict(self, identifiers, input_data):
        raise NotImplementedError

    def input_shape(self, input_name):
        return parse_partial_shape(self.inputs[input_name].get_partial_shape())

    def release(self):
        self.input_feeder.release()
        super().release()

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

    def prepare_model(self, launcher, network_info):
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

        update_mo_params(launcher.config, network_info)
        if 'caffe_model' in network_info:
            model, weights = launcher.convert_model(network_info)
        else:
            model, weights = self.automatic_model_search(network_info)
        return model, weights

    def load_network(self, network, launcher):
        super().load_network(network, launcher)
        self.update_input_output_info()
        self.input_feeder = InputFeeder(
            self.network_info.get('inputs', []), self.inputs, self.input_shape, self.fit_to_input)
        self.infer_request = None

    def load_model(self, network_info, launcher, log=False):
        model_xml, model_bin = self.prepare_model(launcher, network_info)
        self.network = launcher.read_network(str(model_xml), str(model_bin))
        self.load_network(self.network, launcher)
        if log:
            self.print_input_output_info()
        self.infer_request = None

    def update_input_output_info(self):
        if self.default_model_suffix is None:
            return
        config_inputs = self.network_info.get('inputs', [])
        network_with_prefix = next(iter(self.inputs)).startswith(self.default_model_suffix+'_')
        if config_inputs:
            config_with_prefix = config_inputs[0]['name'].startswith(self.default_model_suffix+'_')
            if config_with_prefix == network_with_prefix:
                return
            for c_input in config_inputs:
                c_input['name'] = generate_layer_name(c_input['name'], self.default_model_suffix+'_',
                                                      network_with_prefix)
            self.network_info['inputs'] = config_inputs
        config_outputs = self.network_info['outputs']
        for key, value in config_outputs.items():
            config_with_prefix = value.startswith(self.default_model_suffix+'_')
            if config_with_prefix != network_with_prefix:
                config_outputs[key] = generate_layer_name(value, self.default_model_suffix+'_', network_with_prefix)
        self.network_info['outputs'] = config_outputs


class OpenVINOProposalStage(OVModelMixin, ProposalBaseStage):
    def __init__(
        self, model_info, model_specific_preprocessor, common_preprocessor, launcher, delayed_model_loading=False
    ):
        self.adapter = None
        self.default_model_name = 'mtcnn-p'
        super().__init__(model_info, model_specific_preprocessor, common_preprocessor, launcher, 'pnet',
                         delayed_model_loading)

    def load_network(self, network, launcher):
        super().load_network(network, launcher)
        pnet_outs = self.network_info['outputs']
        pnet_adapter_config = launcher.config.get('adapter', {'type': 'mtcnn_p', **pnet_outs})
        self.adapter = create_adapter(pnet_adapter_config)

    def predict(self, input_blobs, batch_meta, output_callback=None):
        raw_outputs = self._infer(input_blobs, batch_meta)
        if output_callback:
            for out in raw_outputs:
                output_callback(out)
        return raw_outputs


class OpenVINORefineStage(OVModelMixin, RefineBaseStage):
    def __init__(
        self, model_info, model_specific_preprocessor, common_preprocessor, launcher, delayed_model_loading=False
    ):
        self.input_size = 24
        self.include_boundaries = True
        self.default_model_name = 'mtcnn-r'
        super().__init__(model_info, model_specific_preprocessor, common_preprocessor, launcher, 'rnet',
                         delayed_model_loading)

    def predict(self, input_blobs, batch_meta, output_callback=None):
        raw_outputs = self._infer(input_blobs, batch_meta)
        if output_callback:
            batch_size = np.shape(next(iter(input_blobs[0].values())))[0]
            output_callback(transform_for_callback(batch_size, raw_outputs))
        return raw_outputs


class OpenVINOOutputStage(OutputBaseStage, OVModelMixin):
    def __init__(
        self, model_info, model_specific_preprocessor, common_preprocessor, launcher, delayed_model_loading=False
    ):
        self.input_size = 48
        self.include_boundaries = False
        self.default_model_name = 'mtcnn-o'
        super().__init__(model_info, model_specific_preprocessor, common_preprocessor, launcher, 'onet',
                         delayed_model_loading)
