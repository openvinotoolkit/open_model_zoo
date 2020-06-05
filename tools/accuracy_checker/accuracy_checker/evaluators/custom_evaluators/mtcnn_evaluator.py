"""
Copyright (c) 2019 Intel Corporation

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
import pickle
from pathlib import Path
import numpy as np
import cv2

from ..base_evaluator import BaseEvaluator
from ..quantization_model_evaluator import  create_dataset_attributes
from ...adapters import create_adapter, MTCNNPAdapter
from ...launcher import create_launcher, InputFeeder
from ...preprocessor import PreprocessingExecutor
from ...utils import extract_image_representations, read_pickle, contains_any, get_path
from ...config import ConfigError
from ...progress_reporters import ProgressReporter
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

    def preprocess_data(self, batch_input, batch_annotation, previous_stage_prediction, *lrgs, **kwargs):
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
        return model, weights


class DLSDKModelMixin:
    def _infer(self, input_blobs, batch_meta):
        for meta in batch_meta:
            meta['input_shape'] = []
        results = []
        for feed_dict in input_blobs:
            input_shapes = {layer_name: data.shape for layer_name, data in feed_dict.items()}
            self._reshape_input(input_shapes)
            results.append(self.exec_network.infer(feed_dict))
            for meta in batch_meta:
                meta['input_shape'].append(input_shapes)

        return results

    def _reshape_input(self, input_shapes):
        del self.exec_network
        self.network.reshape(input_shapes)
        self.exec_network = self.launcher.ie_core.load_network(self.network, self.launcher.device)

    @property
    def inputs(self):
        has_info = hasattr(self.exec_network, 'input_info')
        if not has_info:
            return self.exec_network.inputs
        return OrderedDict([(name, data.input_data) for name, data in self.exec_network.input_info.items()])

    def release(self):
        self.input_feeder.release()
        del self.network
        del self.exec_network
        self.launcher.release()

    def fit_to_input(self, data, layer_name, layout, precision):
        layer_shape = tuple(self.inputs[layer_name].shape)
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
            print_info('{} - Found model: {}'.format(self.default_model_name, model))
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
            print_info('{} - Found weights: {}'.format(self.default_model_name, weights))
        return model, weights

    def load_network(self, network, launcher, model_prefix):
        self.network = network
        self.exec_network = launcher.ie_core.load_network(network, launcher.device)
        self.update_input_output_info(model_prefix)
        self.input_feeder = InputFeeder(self.model_info.get('inputs', []), self.inputs, self.fit_to_input)

    def load_model(self, network_info, launcher, model_prefix=None, log=False):
        self.network = launcher.read_network(str(network_info['model']), str(network_info['weights']))
        self.exec_network = launcher.ie_core.load_network(self.network, launcher.device)
        self.launcher = launcher
        self.update_input_output_info(model_prefix)
        self.input_feeder = InputFeeder(self.model_info.get('inputs', []), self.inputs, self.fit_to_input)
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
            print_info('\tshape {}\n'.format(input_info.shape))
        print_info('{} - Output info'.format(self.default_model_name))
        for name, output_info in network_outputs.items():
            print_info('\tLayer name: {}'.format(name))
            print_info('\tprecision: {}'.format(output_info.precision))
            print_info('\tshape: {}\n'.format(output_info.shape))

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
        self.input_feeder = InputFeeder(model_info.get('inputs', []), self.inputs, self.fit_to_input)
        pnet_outs = model_info['outputs']
        pnet_adapter_config = launcher.config.get('adapter', {'type': 'mtcnn_p', **pnet_outs})
        pnet_adapter_config.update({'regions_format': 'hw'})
        self.adapter = create_adapter(pnet_adapter_config)


class CaffeRefineStage(CaffeModelMixin, RefineBaseStage):
    def __init__(self, model_info, model_specific_preprocessor, common_preprocessor, launcher, *args, **kwargs):
        super().__init__(model_info, model_specific_preprocessor, common_preprocessor)
        self.net = launcher.create_network(self.model_info['model'], self.model_info['weights'])
        self.input_feeder = InputFeeder(model_info.get('inputs', []), self.inputs, self.fit_to_input)


class CaffeOutputStage(CaffeModelMixin, OutputBaseStage):
    def __init__(self, model_info, model_specific_preprocessor, common_preprocessor, launcher):
        super().__init__(model_info, model_specific_preprocessor, common_preprocessor)
        self.net = launcher.create_network(self.model_info['model'], self.model_info['weights'])
        self.input_feeder = InputFeeder(model_info.get('inputs', []), self.inputs, self.fit_to_input)


class DLSDKProposalStage(DLSDKModelMixin, ProposalBaseStage):
    def __init__(
            self, model_info, model_specific_preprocessor, common_preprocessor, launcher, delayed_model_loading=False
    ):
        super().__init__(model_info, model_specific_preprocessor, common_preprocessor)
        self.adapter = None
        if not delayed_model_loading:
            model_xml, model_bin = self.prepare_model(launcher)
            self.load_model({'model': model_xml, 'weights': model_bin}, launcher, 'pnet_', log=True)
            pnet_outs = model_info['outputs']
            pnet_adapter_config = launcher.config.get('adapter', {'type': 'mtcnn_p', **pnet_outs})
            # pnet_adapter_config.update({'regions_format': 'hw'})
            self.adapter = create_adapter(pnet_adapter_config)

    def load_network(self, network, launcher, model_prefix):
        self.network = network
        self.exec_network = launcher.ie_core.load_network(network, launcher.device)
        self.update_input_output_info(model_prefix)
        self.input_feeder = InputFeeder(self.model_info.get('inputs', []), self.inputs, self.fit_to_input)
        pnet_outs = self.model_info['outputs']
        pnet_adapter_config = launcher.config.get('adapter', {'type': 'mtcnn_p', **pnet_outs})
        self.adapter = create_adapter(pnet_adapter_config)

    def load_model(self, network_info, launcher, model_prefix=None, log=False):
        self.network = launcher.read_network(str(network_info['model']), str(network_info['weights']))
        self.exec_network = launcher.ie_core.load_network(self.network, launcher.device)
        self.launcher = launcher
        self.update_input_output_info(model_prefix)
        self.input_feeder = InputFeeder(self.model_info.get('inputs', []), self.inputs, self.fit_to_input)
        pnet_outs = self.model_info['outputs']
        pnet_adapter_config = launcher.config.get('adapter', {'type': 'mtcnn_p', **pnet_outs})
        self.adapter = create_adapter(pnet_adapter_config)
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
                else:
                    box_outs[layer_name] = np.expand_dims(data[i], axis=0)
            output_per_box.append(box_outs)

        return output_per_box


class DLSDKOutputStage(DLSDKModelMixin, OutputBaseStage):
    def __init__(
            self, model_info, model_specific_preprocessor, common_preprocessor, launcher, delayed_model_loading=False
    ):
        super().__init__(model_info, model_specific_preprocessor, common_preprocessor)
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
                else:
                    box_outs[layer_name] = np.expand_dims(data[i], axis=0)
            output_per_box.append(box_outs)

        return output_per_box




class MTCNNEvaluator(BaseEvaluator):
    def __init__(
            self, dataset_config, launcher, stages
    ):
        self.dataset_config = dataset_config
        self.stages = stages
        self.launcher = launcher
        self.dataset = None
        self.postprocessor = None
        self.metric_executor = None
        self._annotations, self._predictions, self._metrics_results = [], [], []

    def process_dataset(
            self, subset=None,
            num_images=None,
            check_progress=False,
            dataset_tag='',
            output_callback=None,
            allow_pairwise_subset=False,
            dump_prediction_to_annotation=False,
            **kwargs):
        def no_detections(batch_pred):
            return batch_pred[0].size == 0
        self._prepare_dataset(dataset_tag)
        self._create_subset(subset, num_images, allow_pairwise_subset)
        _progress_reporter = self._prepare_progress_reporter(check_progress, kwargs.get('progress_reporter'))

        for batch_id, (batch_input_ids, batch_annotation, batch_inputs, batch_identifiers) in enumerate(self.dataset):
            batch_prediction = []
            batch_raw_prediction = []
            intermediate_callback = None
            if output_callback:
                intermediate_callback = partial(output_callback,
                                                metrics_result=None,
                                                element_identifiers=batch_identifiers,
                                                dataset_indices=batch_input_ids)
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

            metrics_result = None
            if self.metric_executor:
                metrics_result = self.metric_executor.update_metrics_on_batch(
                    batch_input_ids, batch_annotation, batch_prediction
                )
                if self.metric_executor.need_store_predictions:
                    self._annotations.extend(batch_annotation)
                    self._predictions.extend(batch_prediction)

            if output_callback:
                output_callback(
                    list(self.stages.values())[-1].transform_for_callback(batch_size, batch_raw_prediction),
                    metrics_result=metrics_result,
                    element_identifiers=batch_identifiers,
                    dataset_indices=batch_input_ids
                )
            if _progress_reporter:
                _progress_reporter.update(batch_id, len(batch_prediction))

        if _progress_reporter:
            _progress_reporter.finish()

    def compute_metrics(self, print_results=True, ignore_results_formatting=False):
        if self._metrics_results:
            del self._metrics_results
            self._metrics_results = []

        for result_presenter, evaluated_metric in self.metric_executor.iterate_metrics(
                self._annotations, self._predictions):
            self._metrics_results.append(evaluated_metric)
            if print_results:
                result_presenter.write_result(evaluated_metric, ignore_results_formatting)

        return self._metrics_results

    def extract_metrics_results(self, print_results=True, ignore_results_formatting=False):
        if not self._metrics_results:
            self.compute_metrics(False, ignore_results_formatting)

        result_presenters = self.metric_executor.get_metric_presenters()
        extracted_results, extracted_meta = [], []
        for presenter, metric_result in zip(result_presenters, self._metrics_results):
            result, metadata = presenter.extract_result(metric_result)
            if isinstance(result, list):
                extracted_results.extend(result)
                extracted_meta.extend(metadata)
            else:
                extracted_results.append(result)
                extracted_meta.append(metadata)
            if print_results:
                presenter.write_result(metric_result, ignore_results_formatting)

        return extracted_results, extracted_meta

    def print_metrics_results(self, ignore_results_formatting=False):
        if not self._metrics_results:
            self.compute_metrics(True, ignore_results_formatting)
            return
        result_presenters = self.metrics_executor.get_metric_presenters()
        for presenter, metric_result in zip(result_presenters, self._metrics_results):
            presenter.write_result(metric_result, ignore_results_formatting)

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False):
        dataset_config = config['datasets']
        launcher_config = config['launchers'][0]
        if launcher_config['framework'] == 'dlsdk' and 'devise' not in launcher_config:
            launcher_config['device'] = 'CPU'
        models_info = config['network_info']
        launcher = create_launcher(launcher_config, delayed_model_loading=True)
        stages = build_stages(models_info, [], launcher, config.get('_models'), delayed_model_loading)

        return cls(dataset_config, launcher, stages)

    @staticmethod
    def get_processing_info(config):
        module_specific_params = config.get('module_config')
        model_name = config['name']
        dataset_config = module_specific_params['datasets'][0]
        launcher_config = module_specific_params['launchers'][0]
        return (
            model_name, launcher_config['framework'], launcher_config['device'], launcher_config.get('tags'),
            dataset_config['name']
        )

    def release(self):
        for _, stage in self.stages.items():
            stage.release()
        self.launcher.release()

    def reset(self):
        if self.metric_executor:
            self.metric_executor.reset()
        if hasattr(self, '_annotations'):
            del self._annotations
            del self._predictions
        del self._metrics_results
        self._annotations = []
        self._predictions = []
        self._input_ids = []
        self._metrics_results = []
        if self.dataset:
            self.dataset.reset(self.postprocessor.has_processors)
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

    def get_metrics_attributes(self):
        if not self.metric_executor:
            return {}
        return self.metric_executor.get_metrics_attributes()

    def register_metric(self, metric_config):
        if isinstance(metric_config, str):
            self.metric_executor.register_metric({'type': metric_config})
        elif isinstance(metric_config, dict):
            self.metric_executor.register_metric(metric_config)
        else:
            raise ValueError('Unsupported metric configuration type {}'.format(type(metric_config)))

    def register_postprocessor(self, postprocessing_config):
        pass

    def register_dumped_annotations(self):
        pass

    def select_dataset(self, dataset_tag):
        if self.dataset is not None and isinstance(self.dataset_config, list):
            return
        dataset_attributes = create_dataset_attributes(self.dataset_config, dataset_tag)
        self.dataset, self.metric_executor, preprocessor, self.postprocessor = dataset_attributes
        for _, stage in self.stages.items():
            stage.update_preprocessing(preprocessor)

    @staticmethod
    def _create_progress_reporter(check_progress, dataset_size):
        pr_kwargs = {}
        if isinstance(check_progress, int) and not isinstance(check_progress, bool):
            pr_kwargs = {"print_interval": check_progress}

        return ProgressReporter.provide('print', dataset_size, **pr_kwargs)

    def _prepare_dataset(self, dataset_tag=''):
        if self.dataset is None or (dataset_tag and self.dataset.tag != dataset_tag):
            self.select_dataset(dataset_tag)

        if self.dataset.batch is None:
            self.dataset.batch = 1

    def _create_subset(self, subset=None, num_images=None, allow_pairwise=False):
        if subset is not None:
            self.dataset.make_subset(ids=subset, accept_pairs=allow_pairwise)
        elif num_images is not None:
            self.dataset.make_subset(end=num_images, accept_pairs=allow_pairwise)

    def _prepare_progress_reporter(self, check_progress, progress_reporter=None):
        if progress_reporter:
            progress_reporter.reset(self.dataset.size)
            return progress_reporter
        return None if not check_progress else self._create_progress_reporter(check_progress, self.dataset.size)


def calibrate_predictions(previous_stage_predictions, out, threshold, outputs_mapping, iou_type=None):
    score = out[0][outputs_mapping['probability_out']][:, 1]
    pass_t = np.where(score > 0.7)[0]
    removed_boxes = [i for i in range(previous_stage_predictions[0].size) if i not in pass_t]
    previous_stage_predictions[0].remove(removed_boxes)
    previous_stage_predictions[0].scores = score[pass_t]
    bboxes = np.c_[
        previous_stage_predictions[0].x_mins, previous_stage_predictions[0].y_mins,
        previous_stage_predictions[0].x_maxs, previous_stage_predictions[0].y_maxs,
        previous_stage_predictions[0].scores
    ]
    mv = out[0][outputs_mapping['region_out']][pass_t]
    if iou_type:
        previous_stage_predictions[0], peek = nms(previous_stage_predictions[0], threshold, iou_type)
        bboxes = np.c_[
            previous_stage_predictions[0].x_mins, previous_stage_predictions[0].y_mins,
            previous_stage_predictions[0].x_maxs, previous_stage_predictions[0].y_maxs,
            previous_stage_predictions[0].scores
        ]
        mv = mv[np.sort(peek).astype(int)]
    bboxes = bbreg(bboxes, mv.T)
    x_mins, y_mins, x_maxs, y_maxs, _ = bboxes.T
    previous_stage_predictions[0].x_mins = x_mins
    previous_stage_predictions[0].y_mins = y_mins
    previous_stage_predictions[0].x_maxs = x_maxs
    previous_stage_predictions[0].y_maxs = y_maxs

    return previous_stage_predictions


def nms(prediction, threshold, iou_type):
    bboxes = np.c_[
        prediction.x_mins, prediction.y_mins,
        prediction.x_maxs, prediction.y_maxs,
        prediction.scores
    ]
    peek = MTCNNPAdapter.nms(bboxes, threshold, iou_type)
    prediction.remove([i for i in range(prediction.size) if i not in peek])

    return prediction, peek


def bbreg(boundingbox, reg):
    reg = reg.T

    # calibrate bounding boxes
    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1

    bb0 = boundingbox[:, 0] + reg[:, 0] * w
    bb1 = boundingbox[:, 1] + reg[:, 1] * h
    bb2 = boundingbox[:, 2] + reg[:, 2] * w
    bb3 = boundingbox[:, 3] + reg[:, 3] * h

    boundingbox[:, 0:4] = np.array([bb0, bb1, bb2, bb3]).T

    return boundingbox


def pad(boxesA, h, w):
    boxes = boxesA.copy()

    tmph = boxes[:, 3] - boxes[:, 1] + 1
    tmpw = boxes[:, 2] - boxes[:, 0] + 1
    numbox = boxes.shape[0]

    dx = np.ones(numbox)
    dy = np.ones(numbox)
    edx = tmpw
    edy = tmph

    x = boxes[:, 0:1][:, 0]
    y = boxes[:, 1:2][:, 0]
    ex = boxes[:, 2:3][:, 0]
    ey = boxes[:, 3:4][:, 0]

    tmp = np.where(ex > w)[0]
    if tmp.shape[0] != 0:
        edx[tmp] = -ex[tmp] + w - 1 + tmpw[tmp]
        ex[tmp] = w - 1

    tmp = np.where(ey > h)[0]
    if tmp.shape[0] != 0:
        edy[tmp] = -ey[tmp] + h - 1 + tmph[tmp]
        ey[tmp] = h - 1

    tmp = np.where(x < 1)[0]
    if tmp.shape[0] != 0:
        dx[tmp] = 2 - x[tmp]
        x[tmp] = np.ones_like(x[tmp])

    tmp = np.where(y < 1)[0]
    if tmp.shape[0] != 0:
        dy[tmp] = 2 - y[tmp]
        y[tmp] = np.ones_like(y[tmp])

    # for python index from 0, while matlab from 1
    dy = np.maximum(0, dy - 1)
    dx = np.maximum(0, dx - 1)
    y = np.maximum(0, y - 1)
    x = np.maximum(0, x - 1)
    edy = np.maximum(0, edy - 1)
    edx = np.maximum(0, edx - 1)
    ey = np.maximum(0, ey - 1)
    ex = np.maximum(0, ex - 1)
    return [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]


def rerec(bboxA):
    w = bboxA[:, 2] - bboxA[:, 0]
    h = bboxA[:, 3] - bboxA[:, 1]
    l = np.maximum(w, h).T

    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
    bboxA[:, 2:4] = bboxA[:, 0:2] + np.repeat([l], 2, axis=0).T

    return bboxA


def cut_roi(image, prediction, dst_size, include_bound=True):
    bboxes = np.c_[
        prediction.x_mins, prediction.y_mins,
        prediction.x_maxs, prediction.y_maxs,
        prediction.scores
    ]
    img = image.data
    bboxes = rerec(bboxes)
    bboxes[:, 0:4] = np.fix(bboxes[:, 0:4])
    dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(bboxes, *img.shape[:2])
    numbox = bboxes.shape[0]
    tempimg = np.zeros((numbox, dst_size, dst_size, 3))
    for k in range(numbox):
        tmp_k_h = int(tmph[k]) + int(include_bound)
        tmp_k_w = int(tmpw[k]) + int(include_bound)
        tmp = np.zeros((tmp_k_h, tmp_k_w, 3))
        tmp_ys = slice(int(dy[k]), int(edy[k]) + 1)
        tmp_xs = slice(int(dx[k]), int(edx[k]) + 1)
        img_ys = slice(int(y[k]), int(ey[k]) + 1)
        img_xs = slice(int(x[k]), int(ex[k]) + 1)
        tmp[tmp_ys, tmp_xs] = img[img_ys, img_xs]
        tempimg[k, :, :, :] = cv2.resize(tmp, (dst_size, dst_size))
    image.data = tempimg

    return image
