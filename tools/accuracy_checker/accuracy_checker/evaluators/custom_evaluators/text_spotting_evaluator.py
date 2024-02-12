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

from functools import partial
from collections import OrderedDict
import numpy as np

from .base_custom_evaluator import BaseCustomEvaluator
from .base_models import BaseCascadeModel, BaseDLSDKModel, BaseOpenVINOModel, create_model
from ...adapters import create_adapter
from ...config import ConfigError
from ...utils import (
    contains_all,
    extract_image_representations,
    parse_partial_shape,
    generate_layer_name,
    softmax,
    postprocess_output_name
)


class TextSpottingEvaluator(BaseCustomEvaluator):
    def __init__(self, dataset_config, launcher, model, orig_config):
        super().__init__(dataset_config, launcher, orig_config)
        self.model = model
        if hasattr(self.model, 'adapter'):
            self.adapter_type = self.model.adapter.__provider__

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, _ = cls.get_dataset_and_launcher_info(config)
        adapter_info = config['adapter']
        meta = {}
        if config.get('max_seq_len'):
            meta.update({'max_seq_len': config['max_seq_len']})
        if config.get('alphabet'):
            meta.update({'alphabet': config['alphabet']})
        if config.get('sos_index'):
            meta.update({'sos_index': config['sos_index']})
        if config.get('eos_index'):
            meta.update({'eos_index': config['eos_index']})
        if config.get('recognizer_confidence_threshold'):
            meta.update({'recognizer_confidence_threshold': config['recognizer_confidence_threshold']})
        model = SequentialModel(
            config.get('network_info', {}), launcher, config.get('_models', []), adapter_info, meta,
            config.get('_model_is_blob'), delayed_model_loading
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


class SequentialModel(BaseCascadeModel):
    def __init__(self, network_info, launcher, models_args, adapter_info, meta, is_blob=None,
                 delayed_model_loading=False):
        super().__init__(network_info, launcher)
        parts = ['detector', 'recognizer_encoder', 'recognizer_decoder']
        network_info = self.fill_part_with_model(network_info, parts, models_args, is_blob, delayed_model_loading)
        if not contains_all(network_info, parts) and not delayed_model_loading:
            raise ConfigError('network_info should contains detector, encoder and decoder fields')
        self._detector_mapping = {
            'dlsdk': DetectorDLSDKModel,
            'openvino': DetectorOVModel
        }
        self._encoder_mapping = {
            'dlsdk': RecognizerEncoderDLSDKModel,
            'openvino': RecognizerEncoderOVModel
        }
        self._decoder_mapping = {
            'dlsdk': RecognizerDecoderDLSDKModel,
            'openvino': RecognizerDecoderOVModel
        }
        detector_info = network_info.get('detector', {})
        detector_info.update({
            'adapter_info': adapter_info
        })
        self.detector = create_model(detector_info, launcher, self._detector_mapping,
                                     'detector', delayed_model_loading)
        self.recognizer_encoder = create_model(network_info.get('recognizer_encoder', {}), launcher,
                                               self._encoder_mapping, 'encoder', delayed_model_loading)
        self.recognizer_decoder = create_model(network_info.get('recognizer_decoder', {}), launcher,
                                               self._decoder_mapping, 'decoder', delayed_model_loading)
        self.max_seq_len = int(meta.get('max_seq_len', 28))
        self.alphabet = meta.get('alphabet', '__abcdefghijklmnopqrstuvwxyz0123456789')
        self.sos_index = int(meta.get('sos_index', 0))
        self.eos_index = int(meta.get('eos_index', 1))
        self.confidence_threshold = float(meta.get('recognizer_confidence_threshold', '0'))
        self.with_prefix = False
        self._part_by_name = {
            'detector': self.detector,
            'recognizer_encoder': self.recognizer_encoder,
            'recognizer_decoder': self.recognizer_decoder
        }
        if not delayed_model_loading:
            self.update_inputs_outputs_info()

    @property
    def adapter(self):
        return self.detector.adapter

    def predict(self, identifiers, input_data, frame_meta=None, callback=None):
        assert len(identifiers) == 1

        detector_outputs = self.detector.predict(identifiers, input_data)
        if isinstance(detector_outputs, tuple):
            detector_outputs, raw_detector_outputs = detector_outputs
        else:
            raw_detector_outputs = detector_outputs
        text_features = detector_outputs[self.detector.text_feats_out]

        texts = []
        for feature in text_features:
            encoder_outputs = self.recognizer_encoder.predict(
                identifiers, {self.recognizer_encoder.input: np.expand_dims(feature, 0)})

            if isinstance(encoder_outputs, tuple):
                encoder_outputs, raw_encoder_outputs = encoder_outputs
            else:
                raw_encoder_outputs = encoder_outputs

            if callback:
                callback(raw_encoder_outputs)

            feature = encoder_outputs[self.recognizer_encoder.output]
            feature = np.reshape(feature, (feature.shape[0], feature.shape[1], -1))
            feature = np.transpose(feature, (0, 2, 1))
            hidden_shape = self.recognizer_decoder.get_hidden_shape(self.recognizer_decoder.model_inputs['prev_hidden'])
            hidden = np.zeros(hidden_shape)
            prev_symbol_index = np.ones((1,)) * self.sos_index

            text = str()

            confidence = 1.0
            for _ in range(self.max_seq_len):
                input_to_decoder = {
                    self.recognizer_decoder.model_inputs['prev_symbol']: prev_symbol_index,
                    self.recognizer_decoder.model_inputs['prev_hidden']: hidden,
                    self.recognizer_decoder.model_inputs['encoder_outputs']: feature}
                decoder_outputs = self.recognizer_decoder.predict(identifiers, input_to_decoder)
                if isinstance(decoder_outputs, tuple):
                    decoder_outputs, raw_decoder_outputs = decoder_outputs
                else:
                    raw_decoder_outputs = decoder_outputs

                if callback:
                    callback(raw_decoder_outputs)
                decoder_output = decoder_outputs[self.recognizer_decoder.model_outputs['symbols_distribution']]
                softmaxed = softmax(decoder_output[0])
                prev_symbol_index = np.argmax(decoder_output, axis=1)
                confidence *= softmaxed[prev_symbol_index]
                if prev_symbol_index == self.eos_index:
                    break
                hidden = decoder_outputs[self.recognizer_decoder.model_outputs['cur_hidden']]
                text += self.alphabet[int(prev_symbol_index)]
            texts.append(text if confidence >= self.confidence_threshold else '')

        texts = np.array(texts)
        detector_outputs['texts'] = texts
        output = self.adapter.process(detector_outputs, identifiers, frame_meta)
        return raw_detector_outputs, output

    def load_model(self, network_list, launcher):
        super().load_model(network_list, launcher)
        self.update_inputs_outputs_info()

    def load_network(self, network_list, launcher):
        super().load_network(network_list, launcher)
        self.update_inputs_outputs_info()

    def update_inputs_outputs_info(self):
        with_prefix = (
            isinstance(self.detector.im_data_name, str) and self.detector.im_data_name.startswith('detector_')
        )
        self.adapter.outputs_verified = False
        if hasattr(self.detector, 'outputs'):
            text_feats_out = postprocess_output_name(
                self.detector.text_feats_out, self.detector.outputs,
                additional_mapping=self.detector.additional_output_mapping, raise_error=False)
            if text_feats_out not in self.detector.outputs:
                text_feats_out = postprocess_output_name(
                generate_layer_name(self.detector.text_feats_out, 'detector_', with_prefix),
                self.detector.outputs,
                additional_mapping=self.detector.additional_output_mapping, raise_error=False)
            self.detector.text_feats_out = text_feats_out
            encoder_output = postprocess_output_name(
                self.recognizer_encoder.output, self.recognizer_encoder.outputs,
                additional_mapping=self.recognizer_encoder.additional_output_mapping, raise_error=False
            )
            if encoder_output not in self.recognizer_encoder.outputs:
                encoder_output = postprocess_output_name(
                generate_layer_name(self.recognizer_encoder.output, 'recognizer_encoder_', with_prefix),
                self.recognizer_encoder.outputs,
                additional_mapping=self.recognizer_encoder.additional_output_mapping, raise_error=False
            )
            self.recognizer_encoder.output = encoder_output
            for out, out_value in self.recognizer_decoder.model_outputs.items():
                output = postprocess_output_name(
                    out_value, self.recognizer_decoder.outputs,
                    additional_mapping=self.recognizer_decoder.additional_output_mapping, raise_error=False
                )
                if output not in self.recognizer_decoder.outputs:
                    output = postprocess_output_name(
                    generate_layer_name(out_value, 'recognizer_decoder_', with_prefix),
                    self.recognizer_decoder.outputs,
                    additional_mapping=self.recognizer_decoder.additional_output_mapping, raise_error=False
                )
                self.recognizer_decoder.model_outputs[out] = output
        if with_prefix != self.with_prefix:
            self.recognizer_encoder.input = generate_layer_name(
                self.recognizer_encoder.input, 'recognizer_encoder_', with_prefix
            )
            recognizer_decoder_inputs = {
                key: generate_layer_name(value, 'recognizer_decoder_', with_prefix)
                for key, value in self.recognizer_decoder.model_inputs.items()
            }
            self.recognizer_decoder.model_inputs = recognizer_decoder_inputs
        self.with_prefix = with_prefix


class DetectorDLSDKModel(BaseDLSDKModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.im_info_name = None
        self.im_data_name = None
        self.adapter = create_adapter(network_info['adapter_info'])
        super().__init__(network_info, launcher, suffix, delayed_model_loading)

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
            self._reshape_input({k: v.shape for k, v in input_data.items()})

        output = self.exec_network.infer(input_data)

        return output

    def fit_to_input(self, input_data):
        input_data = np.transpose(input_data, (0, 3, 1, 2))
        has_info = hasattr(self.exec_network, 'input_info')
        input_info = (
            self.exec_network.input_info[self.im_data_name].input_data
            if has_info else self.exec_network.inputs[self.im_data_name]
        )
        input_data = input_data.reshape(input_info.shape)

        return input_data

    def set_input_and_output(self):
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


class DetectorOVModel(BaseOpenVINOModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.im_info_name = None
        self.im_data_name = None
        self.adapter_info = network_info['adapter_info']
        super().__init__(network_info, launcher, suffix, delayed_model_loading)

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
            self._reshape_input({k: v.shape for k, v in input_data.items()})

        output = self.infer(input_data, raw_results=True)

        return output

    def fit_to_input(self, input_data):
        input_data = np.transpose(input_data, (0, 3, 1, 2))

        return input_data

    def set_input_and_output(self):
        self.im_data_name = [
            x for x in self.inputs if len(parse_partial_shape(self.inputs[x].get_partial_shape())) == 4][0]
        self.im_info_name = [
            x for x in self.inputs if len(parse_partial_shape(self.inputs[x].get_partial_shape())) == 2
        ]
        if self.im_info_name:
            self.im_info_name = self.im_info_name[0]
            self.text_feats_out = 'text_features'
        else:
            self.text_feats_out = 'text_features'
        self.adapter = create_adapter(self.adapter_info, additional_output_mapping=self.additional_output_mapping)


class RecognizerDLSDKModel(BaseDLSDKModel):
    def predict(self, identifiers, input_data):
        if not self.is_dynamic and self.dynamic_inputs:
            self._reshape_input({k: v.shape for k, v in input_data.items()})
        return self.exec_network.infer(input_data)

    def get_hidden_shape(self, name):
        has_info = hasattr(self.exec_network, 'input_info')
        if has_info:
            hidden_shape = self.exec_network.input_info[name].input_data.shape
        else:
            hidden_shape = self.exec_network.inputs[name].shape
        return hidden_shape


class RecognizerOVModel(BaseOpenVINOModel):
    def predict(self, identifiers, input_data):
        if not self.is_dynamic and self.dynamic_inputs:
            self._reshape_input({k: v.shape for k, v in input_data.items()})
        return self.infer(input_data, raw_results=True)

    def get_hidden_shape(self, name):
        return parse_partial_shape(self.inputs[name].get_partial_shape())


class RecognizerEncoderDLSDKModel(RecognizerDLSDKModel):
    def __init__(self, network_info, launcher, suffix, delayed_model_loading=False):
        self.input = 'input'
        self.output = 'output'
        super().__init__(network_info, launcher, suffix, delayed_model_loading)


class RecognizerEncoderOVModel(RecognizerOVModel):
    def __init__(self, network_info, launcher, suffix, delayed_model_loading=False):
        self.input = 'input'
        self.output = 'output/sink_port_0'
        super().__init__(network_info, launcher, suffix, delayed_model_loading)


class RecognizerDecoderDLSDKModel(RecognizerDLSDKModel):
    def __init__(self, network_info, launcher, suffix, delayed_model_loading=False):
        self.model_inputs = network_info['inputs']
        self.model_outputs = network_info['outputs']
        super().__init__(network_info, launcher, suffix, delayed_model_loading)


class RecognizerDecoderOVModel(RecognizerOVModel):
    def __init__(self, network_info, launcher, suffix, delayed_model_loading=False):
        def preprocess_out(out_name):
            if not out_name.endswith('/sink_port_0'):
                return out_name + '/sink_port_0'
            return out_name
        self.model_inputs = network_info['inputs']
        self.model_outputs = {out: preprocess_out(name) for out, name in network_info['outputs'].items()}
        super().__init__(network_info, launcher, suffix, delayed_model_loading)
