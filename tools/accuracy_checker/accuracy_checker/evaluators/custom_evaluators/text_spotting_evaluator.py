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
from pathlib import Path
from functools import partial
from collections import OrderedDict
import numpy as np

from ..base_evaluator import BaseEvaluator
from ..quantization_model_evaluator import create_dataset_attributes
from ...adapters import create_adapter
from ...config import ConfigError
from ...launcher import create_launcher
from ...utils import contains_all, extract_image_representations, get_path
from ...progress_reporters import ProgressReporter
from ...logging import print_info


class TextSpottingEvaluator(BaseEvaluator):
    def __init__(self, dataset_config, launcher, model):
        self.dataset_config = dataset_config
        self.preprocessing_executor = None
        self.preprocessor = None
        self.dataset = None
        self.postprocessor = None
        self.metric_executor = None
        self.launcher = launcher
        self.model = model
        self._metrics_results = []

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False):
        dataset_config = config['datasets']
        launcher_config = config['launchers'][0]
        if launcher_config['framework'] == 'dlsdk' and 'devise' not in launcher_config:
            launcher_config['device'] = 'CPU'

        launcher = create_launcher(launcher_config, delayed_model_loading=True)
        model = SequentialModel(
            config.get('network_info', {}), launcher, config.get('_models', []), config.get('_model_is_blob'),
            delayed_model_loading
        )
        return cls(dataset_config, launcher, model)

    def process_dataset(
            self, subset=None,
            num_images=None,
            check_progress=False,
            dataset_tag='',
            output_callback=None,
            allow_pairwise_subset=False,
            dump_prediction_to_annotation=False,
            **kwargs):
        self._prepare_dataset(dataset_tag)
        self._create_subset(subset, num_images, allow_pairwise_subset)
        self._annotations, self._predictions = [], []
        if 'progress_reporter' in kwargs:
            _progress_reporter = kwargs['progress_reporter']
            _progress_reporter.reset(self.dataset.size)
        else:
            _progress_reporter = None if not check_progress else self._create_progress_reporter(
                check_progress, self.dataset.size
            )
        for batch_id, (batch_input_ids, batch_annotation, batch_inputs, batch_identifiers) in enumerate(self.dataset):
            batch_inputs = self.preprocessor.process(batch_inputs, batch_annotation)
            batch_data, batch_meta = extract_image_representations(batch_inputs)
            temporal_output_callback = None
            if output_callback:
                temporal_output_callback = partial(output_callback,
                                                   metrics_result=None,
                                                   element_identifiers=batch_identifiers,
                                                   dataset_indices=batch_input_ids)

            batch_raw_prediction, batch_prediction = self.model.predict(
                batch_identifiers, batch_data, batch_meta, callback=temporal_output_callback
            )
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
                    batch_raw_prediction,
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
                self._annotations, self._predictions
        ):
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
        result_presenters = self.metric_executor.get_metric_presenters()
        for presenter, metric_result in zip(result_presenters, self._metrics_results):
            presenter.write_result(metric_result, ignore_results_formatting)

    def release(self):
        self.model.release()
        self.launcher.release()

    def reset(self):
        if self.metric_executor:
            self.metric_executor.reset()
        if hasattr(self, '_annotations'):
            del self._annotations
            del self._predictions
            del self._input_ids
        del self._metrics_results
        self._annotations = []
        self._predictions = []
        self._input_ids = []
        self._metrics_results = []
        if self.dataset:
            self.dataset.reset(self.postprocessor.has_processors)

    @staticmethod
    def get_processing_info(config):
        module_specific_params = config.get('module_config')
        model_name = config['name']
        dataset_config = module_specific_params['datasets'][0]
        launcher_config = module_specific_params['launchers'][0]
        return (
            model_name, launcher_config['framework'], launcher_config['device'],
            launcher_config.get('tags'),
            dataset_config['name']
        )

    def load_network(self, network=None):
        self.model.load_network(network, self.launcher)

    def load_network_from_ir(self, models_dict):
        self.model.load_model(models_dict, self.launcher)

    def get_network(self):
        return self.model.get_network()

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
        self.dataset, self.metric_executor, self.preprocessor, self.postprocessor = dataset_attributes

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


class BaseModel:
    def __init__(self, network_info, launcher, default_model_suffix, delayed_model_loading=False):
        self.default_model_suffix = default_model_suffix
        self.network_info = network_info

    def predict(self, identifiers, input_data):
        raise NotImplementedError

    def release(self):
        pass

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
            print_info('{} - Found model: {}'.format(self.default_model_suffix, model))
        if model.suffix == '.blob':
            return model, None
        weights = get_path(network_info.get('weights', model.parent / model.name.replace('xml', 'bin')))
        print_info('{} - Found weights: {}'.format(self.default_model_suffix, weights))

        return model, weights

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
            print_info('\tshape {}\n'.format(input_info.shape))
        print_info('{} - Output info'.format(self.default_model_suffix))
        for name, output_info in network_outputs.items():
            print_info('\tLayer name: {}'.format(name))
            print_info('\tprecision: {}'.format(output_info.precision))
            print_info('\tshape: {}\n'.format(output_info.shape))

    def load_network(self, network, launcher):
        self.network = network
        self.exec_network = launcher.ie_core.load_network(network, launcher.device)

    def get_network(self):
        return self.network


def create_detector(model_config, launcher, delayed_model_loading=False):
    launcher_model_mapping = {
        'dlsdk': DetectorDLSDKModel
    }
    framework = launcher.config['framework']
    model_class = launcher_model_mapping.get(framework)
    if not model_class:
        raise ValueError('model for framework {} is not supported'.format(framework))
    return model_class(model_config, launcher, delayed_model_loading)


def create_recognizer(model_config, launcher, suffix, delayed_model_loading=False):
    launcher_model_mapping = {
        'dlsdk': RecognizerDLSDKModel
    }
    framework = launcher.config['framework']
    model_class = launcher_model_mapping.get(framework)
    if not model_class:
        raise ValueError('model for framework {} is not supported'.format(framework))
    return model_class(model_config, launcher, suffix, delayed_model_loading)


class SequentialModel:
    def __init__(self, network_info, launcher, models_args, is_blob=None, delayed_model_loading=False):
        if not delayed_model_loading:
            detector = network_info.get('detector', {})
            recognizer_encoder = network_info.get('recognizer_encoder', {})
            recognizer_decoder = network_info.get('recognizer_decoder', {})
            if 'model' not in detector:
                detector['model'] = models_args[0]
                detector['_model_is_blob'] = is_blob
            if 'model' not in recognizer_encoder:
                recognizer_encoder['model'] = models_args[1 if len(models_args) > 1 else 0]
                recognizer_encoder['_model_is_blob'] = is_blob
            if 'model' not in recognizer_decoder:
                recognizer_decoder['model'] = models_args[2 if len(models_args) > 2 else 0]
                recognizer_decoder['_model_is_blob'] = is_blob
            network_info.update({
                'detector': detector,
                'recognizer_encoder': recognizer_encoder,
                'recognizer_decoder': recognizer_decoder
            })
            if not contains_all(network_info, ['detector', 'recognizer_encoder', 'recognizer_decoder']):
                raise ConfigError('network_info should contains detector, encoder and decoder fields')
        self.detector = create_detector(network_info.get('detector', {}), launcher, delayed_model_loading)
        self.recognizer_encoder = create_recognizer(
            network_info.get('recognizer_encoder', {}), launcher, 'encoder', delayed_model_loading
        )
        self.recognizer_decoder = create_recognizer(
            network_info.get('recognizer_decoder', {}), launcher, 'decoder', delayed_model_loading
        )
        self.recognizer_decoder_inputs = network_info['recognizer_decoder_inputs']
        self.recognizer_decoder_outputs = network_info['recognizer_decoder_outputs']
        self.recognizer_encoder_input = 'input'
        self.recognizer_encoder_output = 'output'
        self.max_seq_len = int(network_info['max_seq_len'])
        self.adapter = create_adapter(network_info['adapter'])
        self.alphabet = network_info['alphabet']
        self.sos_index = int(network_info['sos_index'])
        self.eos_index = int(network_info['eos_index'])
        self.with_prefix = False
        self._part_by_name = {
            'detector': self.detector,
            'recognizer_encoder': self.recognizer_encoder,
            'recognizer_decoder': self.recognizer_decoder
        }

    def predict(self, identifiers, input_data, frame_meta, callback):
        assert len(identifiers) == 1

        detector_outputs = self.detector.predict(identifiers, input_data)
        text_features = detector_outputs[self.detector.text_feats_out]

        texts = []
        decoder_exec_net = self.recognizer_decoder.exec_network
        has_info = hasattr(decoder_exec_net, 'input_info')
        for feature in text_features:
            encoder_outputs = self.recognizer_encoder.predict(identifiers, {self.recognizer_encoder_input: feature})
            if callback:
                callback(encoder_outputs)

            feature = encoder_outputs[self.recognizer_encoder_output]
            feature = np.reshape(feature, (feature.shape[0], feature.shape[1], -1))
            feature = np.transpose(feature, (0, 2, 1))
            if has_info:
                hidden_shape = decoder_exec_net.input_info[
                    self.recognizer_decoder_inputs['prev_hidden']
                ].input_data.shape
            else:
                hidden_shape = decoder_exec_net.inputs[self.recognizer_decoder_inputs['prev_hidden']].shape
            hidden = np.zeros(hidden_shape)
            prev_symbol_index = np.ones((1,)) * self.sos_index

            text = str()

            for _ in range(self.max_seq_len):
                input_to_decoder = {
                    self.recognizer_decoder_inputs['prev_symbol']: prev_symbol_index,
                    self.recognizer_decoder_inputs['prev_hidden']: hidden,
                    self.recognizer_decoder_inputs['encoder_outputs']: feature}
                decoder_outputs = self.recognizer_decoder.predict(identifiers, input_to_decoder)
                if callback:
                    callback(decoder_outputs)
                coder_output = decoder_outputs[self.recognizer_decoder_outputs['symbols_distribution']]
                prev_symbol_index = np.argmax(coder_output, axis=1)
                if prev_symbol_index == self.eos_index:
                    break
                hidden = decoder_outputs[self.recognizer_decoder_outputs['cur_hidden']]
                text += self.alphabet[int(prev_symbol_index)]
            texts.append(text)

        texts = np.array(texts)

        detector_outputs['texts'] = texts
        output = self.adapter.process(detector_outputs, identifiers, frame_meta)
        return detector_outputs, output

    def release(self):
        self.detector.release()
        self.recognizer_encoder.release()
        self.recognizer_decoder.release()

    def load_model(self, network_list, launcher):
        for network_dict in network_list:
            self._part_by_name[network_dict['name']].load_model(network_dict, launcher)
        self.update_inputs_outputs_info()

    def load_network(self, network_list, launcher):
        for network_dict in network_list:
            self._part_by_name[network_dict['name']].load_network(network_dict['model'], launcher)
        self.update_inputs_outputs_info()

    def get_network(self):
        return [
            {'name': 'detector', 'model': self.detector.get_network()},
            {'name': 'recognizer_encoder', 'model': self.recognizer_encoder.get_network()},
            {'name': 'recognizer_decoder', 'model': self.recognizer_decoder.get_network()}
            ]

    def update_inputs_outputs_info(self):
        def generate_name(prefix, with_prefix, layer_name):
            return prefix + layer_name if with_prefix else layer_name.split(prefix)[-1]

        with_prefix = (
            isinstance(self.detector.im_data_name, str) and self.detector.im_data_name.startswith('detector_')
        )
        if with_prefix != self.with_prefix:
            self.detector.text_feats_out = generate_name('detector_', with_prefix, self.detector.text_feats_out)
            self.adapter.classes_out = generate_name('detector_', with_prefix, self.adapter.classes_out)
            self.adapter.scores_out = generate_name('detector_', with_prefix, self.adapter.scores_out)
            self.adapter.boxes_out = generate_name('detector_', with_prefix, self.adapter.boxes_out)
            self.adapter.raw_masks_out = generate_name('detector_', with_prefix, self.adapter.raw_masks_out)
            self.recognizer_encoder_input = generate_name(
                'recognizer_encoder_', with_prefix, self.recognizer_encoder_input
            )
            self.recognizer_encoder_output = generate_name(
                'recognizer_encoder_', with_prefix, self.recognizer_encoder_output
            )
            recognizer_decoder_inputs = {
                key: generate_name('recognizer_decoder_', with_prefix, value)
                for key, value in self.recognizer_decoder_inputs.items()
            }
            recognizer_decoder_outputs = {
                key: generate_name('recognizer_decoder_', with_prefix, value)
                for key, value in self.recognizer_decoder_outputs.items()
            }
            self.recognizer_decoder_inputs = recognizer_decoder_inputs
            self.recognizer_decoder_outputs = recognizer_decoder_outputs
        self.with_prefix = with_prefix


class DetectorDLSDKModel(BaseModel):
    def __init__(self, network_info, launcher, delayed_model_loading=False):
        super().__init__(network_info, launcher, 'detector')
        self.im_info_name = None
        self.im_data_name = None
        if not delayed_model_loading:
            self.load_model(network_info, launcher, log=True)
            has_info = hasattr(self.exec_network, 'input_info')
            input_info = (
                OrderedDict([(name, data.input_data) for name, data in self.exec_network.input_info.items()])
                if has_info else self.exec_network.inputs
            )
            self.im_info_name = [x for x in input_info if len(input_info[x].shape) == 2][0]
            self.im_data_name = [x for x in input_info if len(input_info[x].shape) == 4][0]
        self.text_feats_out = 'text_features'

    def predict(self, identifiers, input_data):

        input_data = np.array(input_data)
        assert len(input_data.shape) == 4
        assert input_data.shape[0] == 1

        input_data = {self.im_data_name: self.fit_to_input(input_data),
                      self.im_info_name: np.array(
                          [[input_data.shape[1], input_data.shape[2], 1.0]])}

        output = self.exec_network.infer(input_data)

        return output

    def release(self):
        del self.network
        del self.exec_network

    def fit_to_input(self, input_data):
        input_data = np.transpose(input_data, (0, 3, 1, 2))
        has_info = hasattr(self.exec_network, 'input_info')
        input_info = (
            self.exec_network.input_info[self.im_data_name].input_data
            if has_info else self.exec_network.inputs[self.im_data_name]
        )
        input_data = input_data.reshape(input_info.shape)

        return input_data

    def load_model(self, network_info, launcher, log=False):
        model, weights = self.automatic_model_search(network_info)
        if weights is not None:
            self.network = launcher.read_network(str(model), str(weights))
            self.exec_network = launcher.ie_core.load_network(self.network, launcher.device)
        else:
            self.exec_network = launcher.ie_core.import_network(str(model))
        has_info = hasattr(self.exec_network, 'input_info')
        input_info = (
            OrderedDict([(name, data.input_data) for name, data in self.exec_network.input_info.items()])
            if has_info else self.exec_network.inputs
        )
        self.im_info_name = [x for x in input_info if len(input_info[x].shape) == 2][0]
        self.im_data_name = [x for x in input_info if len(input_info[x].shape) == 4][0]
        if log:
            self.print_input_output_info()


class RecognizerDLSDKModel(BaseModel):
    def __init__(self, network_info, launcher, suffix, delayed_model_loading=False):
        super().__init__(network_info, launcher, suffix)
        if not delayed_model_loading:
            self.load_model(network_info, launcher, log=True)

    def predict(self, identifiers, input_data):
        return self.exec_network.infer(input_data)

    def release(self):
        del self.network
        del self.exec_network

    def load_model(self, network_info, launcher, log=False):
        model, weights = self.automatic_model_search(network_info)
        if weights is not None:
            self.network = launcher.read_network(str(model), str(weights))
            self.exec_network = launcher.ie_core.load_network(self.network, launcher.device)
        else:
            self.exec_network = launcher.ie_core.import_network(str(model))
        if log:
            self.print_input_output_info()
