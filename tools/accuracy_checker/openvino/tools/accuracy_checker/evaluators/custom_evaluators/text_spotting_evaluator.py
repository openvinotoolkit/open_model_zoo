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
from functools import partial
from collections import OrderedDict
import numpy as np

from .base_custom_evaluator import BaseCustomEvaluator
from ...adapters import create_adapter
from ...config import ConfigError
from ...utils import contains_all, extract_image_representations, get_path
from ...logging import print_info


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class TextSpottingEvaluator(BaseCustomEvaluator):
    def __init__(self, dataset_config, launcher, model, orig_config):
        super().__init__(dataset_config, launcher, orig_config)
        self.model = model
        self.adapter_type = self.model.adapter.__provider__

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, _ = cls.get_dataset_and_launcher_info(config)
        model = SequentialModel(
            config.get('network_info', {}), launcher, config.get('_models', []), config.get('_model_is_blob'),
            delayed_model_loading
        )
        return cls(dataset_config, launcher, model, orig_config)

    def _process(self, output_callback, calculate_metrics, progress_reporter, metric_config, csv_file):
        for batch_id, (batch_input_ids, batch_annotation, batch_inputs, batch_identifiers) in enumerate(self.dataset):
            batch_inputs = self.preprocessor.process(batch_inputs, batch_annotation)
            batch_data, batch_meta = extract_image_representations(batch_inputs)
            temporal_output_callback = None
            if output_callback:
                temporal_output_callback = partial(output_callback, metrics_result=None,
                                                   element_identifiers=batch_identifiers,
                                                   dataset_indices=batch_input_ids)
            batch_raw_prediction, batch_prediction = self.model.predict(
                batch_identifiers, batch_data, batch_meta, callback=temporal_output_callback
            )
            batch_annotation, batch_prediction = self.postprocessor.process_batch(
                batch_annotation, batch_prediction, batch_meta
            )
            metrics_result = self._get_metrics_result(batch_input_ids, batch_annotation, batch_prediction,
                                                      calculate_metrics)
            if output_callback:
                output_callback(batch_raw_prediction, metrics_result=metrics_result,
                                element_identifiers=batch_identifiers, dataset_indices=batch_input_ids)
            self._update_progress(progress_reporter, metric_config, batch_id, len(batch_prediction), csv_file)


class BaseModel:
    def __init__(self, network_info, launcher, default_model_suffix, delayed_model_loading=False):
        self.is_dynamic = False
        self.default_model_suffix = default_model_suffix
        self.network_info = network_info
        self.launcher = launcher

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
        accepted_suffixes = ['.blob', '.xml']
        if model.suffix not in accepted_suffixes:
            raise ConfigError('Models with following suffixes are allowed: {}'.format(accepted_suffixes))
        print_info('{} - Found model: {}'.format(self.default_model_suffix, model))
        if model.suffix == '.blob':
            return model, None
        weights = get_path(network_info.get('weights', model.parent / model.name.replace('xml', 'bin')))
        accepted_weights_suffixes = ['.bin']
        if weights.suffix not in accepted_weights_suffixes:
            raise ConfigError('Weights with following suffixes are allowed: {}'.format(accepted_weights_suffixes))
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
            print_info('\tshape: {}\n'.format(
                input_info.shape if name not in self.partial_shapes else self.partial_shapes[name]))
        print_info('{} - Output info'.format(self.default_model_suffix))
        for name, output_info in network_outputs.items():
            print_info('\tLayer name: {}'.format(name))
            print_info('\tprecision: {}'.format(output_info.precision))
            print_info('\tshape: {}\n'.format(
                output_info.shape if name not in self.partial_shapes else self.partial_shapes[name]))

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

    def reshape_net(self, shape):
        if self.is_dynamic:
            return
        if hasattr(self, 'exec_network') and self.exec_network is not None:
            del self.exec_network
        self.network.reshape(shape)
        self.dynamic_inputs, self.partial_shapes = self.launcher.get_dynamic_inputs(self.network)
        if not self.is_dynamic and self.dynamic_inputs:
            return
        self.exec_network = self.launcher.load_network(self.network, self.launcher.device)

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
        self.confidence_threshold = float(network_info.get('recognizer_confidence_threshold', '0'))
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

            confidence = 1.0
            for _ in range(self.max_seq_len):
                input_to_decoder = {
                    self.recognizer_decoder_inputs['prev_symbol']: prev_symbol_index,
                    self.recognizer_decoder_inputs['prev_hidden']: hidden,
                    self.recognizer_decoder_inputs['encoder_outputs']: feature}
                decoder_outputs = self.recognizer_decoder.predict(identifiers, input_to_decoder)
                if callback:
                    callback(decoder_outputs)
                decoder_output = decoder_outputs[self.recognizer_decoder_outputs['symbols_distribution']]
                softmaxed = softmax(decoder_output[0])
                prev_symbol_index = np.argmax(decoder_output, axis=1)
                confidence *= softmaxed[prev_symbol_index]
                if prev_symbol_index == self.eos_index:
                    break
                hidden = decoder_outputs[self.recognizer_decoder_outputs['cur_hidden']]
                text += self.alphabet[int(prev_symbol_index)]
            texts.append(text if confidence >= self.confidence_threshold else '')

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
            self.adapter.classes_out = generate_name('detector_', with_prefix, self.adapter.classes_out)
            if self.adapter.scores_out is not None:
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
            if self.exec_network is not None:
                has_info = hasattr(self.exec_network, 'input_info')
                input_info = (
                    OrderedDict([(name, data.input_data) for name, data in self.exec_network.input_info.items()])
                    if has_info else self.exec_network.inputs
                )
            else:
                has_info = hasattr(self.network, 'input_info')
                input_info = (
                    OrderedDict([(name, data.input_data) for name, data in self.network.input_info.items()])
                    if has_info else self.network.inputs
                )
            self.im_info_name = [x for x in input_info if len(input_info[x].shape) == 2]
            self.im_data_name = [x for x in input_info if len(input_info[x].shape) == 4][0]
            if self.im_info_name:
                self.im_info_name = self.im_info_name[0]
                self.text_feats_out = 'detector_text_features' if self.im_data_name.startswith(
                    'detector_') else 'text_features'
            else:
                self.text_feats_out = 'detector_text_features.0' if self.im_data_name.startswith(
                    'detector_') else 'text_features.0'

    def predict(self, identifiers, input_data):
        input_data = np.array(input_data)
        assert len(input_data.shape) == 4
        assert input_data.shape[0] == 1

        if self.im_info_name:
            input_data = {self.im_data_name: self.fit_to_input(input_data),
                          self.im_info_name: np.array([[input_data.shape[1], input_data.shape[2], 1.0]])}
        else:
            input_data = {self.im_data_name: self.fit_to_input(input_data)}
        if not self.is_dynamic and self.dynamic_inputs:
            self.reshape_net({k: v.shape for k, v in input_data.items()})

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
            self.load_network(self.network, launcher)
        else:
            self.exec_network = launcher.ie_core.import_network(str(model))
        if self.exec_network:
            has_info = hasattr(self.exec_network, 'input_info')
            input_info = (
                OrderedDict([(name, data.input_data) for name, data in self.exec_network.input_info.items()])
                if has_info else self.exec_network.inputs
            )
        else:
            has_info = hasattr(self.network, 'input_info')
            input_info = (
                OrderedDict([(name, data.input_data) for name, data in self.network.input_info.items()])
                if has_info else self.network.inputs
            )
        self.im_data_name = [x for x in input_info if len(input_info[x].shape) == 4][0]
        self.im_info_name = [x for x in input_info if len(input_info[x].shape) == 2]
        if self.im_info_name:
            self.im_info_name = self.im_info_name[0]
            self.text_feats_out = 'detector_text_features' if self.im_data_name.startswith(
                'detector_') else 'text_features'
        else:
            self.text_feats_out = 'detector_text_features.0' if self.im_data_name.startswith(
                'detector_') else 'text_features.0'
        if log:
            self.print_input_output_info()


class RecognizerDLSDKModel(BaseModel):
    def __init__(self, network_info, launcher, suffix, delayed_model_loading=False):
        super().__init__(network_info, launcher, suffix)
        if not delayed_model_loading:
            self.load_model(network_info, launcher, log=True)

    def predict(self, identifiers, input_data):
        if not self.is_dynamic and self.dynamic_inputs:
            self.reshape_net({k: v.shape for k, v in input_data.items()})
        return self.exec_network.infer(input_data)

    def release(self):
        del self.network
        del self.exec_network

    def load_model(self, network_info, launcher, log=False):
        model, weights = self.automatic_model_search(network_info)
        if weights is not None:
            self.network = launcher.read_network(str(model), str(weights))
            self.load_network(self.network, launcher)
        else:
            self.exec_network = launcher.ie_core.import_network(str(model))
        if log:
            self.print_input_output_info()
