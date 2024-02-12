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
import pickle  # nosec B403  # disable import-pickle check
from functools import partial
from collections import OrderedDict


from .base_custom_evaluator import BaseCustomEvaluator
from .base_models import (
    BaseCascadeModel, BaseDLSDKModel, BaseONNXModel, BaseOpenVINOModel,
    create_model, create_encoder)
from ...adapters import create_adapter
from ...config import ConfigError
from ...utils import contains_all, extract_image_representations, read_pickle


class AutomaticSpeechRecognitionEvaluator(BaseCustomEvaluator):
    def __init__(self, dataset_config, launcher, model, orig_config):
        super().__init__(dataset_config, launcher, orig_config)
        self.model = model
        if hasattr(self.model, 'adapter'):
            self.adapter_type = self.model.adapter.__provider__

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, _ = cls.get_dataset_and_launcher_info(config)
        model = ASRModel(
            config.get('network_info', {}), launcher, config.get('_models', []), config.get('_model_is_blob'),
            delayed_model_loading
        )
        return cls(dataset_config, launcher, model, orig_config)

    def _process(self, output_callback, calculate_metrics, progress_reporter, metric_config, csv_file):
        for batch_id, (batch_input_ids, batch_annotation, batch_inputs, batch_identifiers) in enumerate(self.dataset):
            batch_inputs = self.preprocessor.process(batch_inputs, batch_annotation)
            batch_inputs_extr, _ = extract_image_representations(batch_inputs)
            encoder_callback = None
            if output_callback:
                encoder_callback = partial(output_callback, metrics_result=None, element_identifiers=batch_identifiers,
                                           dataset_indices=batch_input_ids)
            batch_raw_prediction, batch_prediction = self.model.predict(
                batch_identifiers, batch_inputs_extr, encoder_callback=encoder_callback
            )
            metrics_result = self._get_metrics_result(batch_input_ids, batch_annotation, batch_prediction,
                                                      calculate_metrics)
            if output_callback:
                output_callback(batch_raw_prediction[0], metrics_result=metrics_result,
                                element_identifiers=batch_identifiers, dataset_indices=batch_input_ids)
            self._update_progress(progress_reporter, metric_config, batch_id, len(batch_prediction), csv_file)


class ASRModel(BaseCascadeModel):
    def __init__(self, network_info, launcher, models_args, is_blob, delayed_model_loading=False):
        super().__init__(network_info, launcher)
        parts = ['encoder', 'decoder']
        network_info = self.fill_part_with_model(network_info, parts, models_args, is_blob, delayed_model_loading)
        if not contains_all(network_info, parts) and not delayed_model_loading:
            raise ConfigError('network_info should contain encoder and decoder fields')
        self._decoder_mapping = {
            'dlsdk': DecoderDLSDKModel,
            'openvino': DecoderOVModel,
            'onnx_runtime': DecoderONNXModel
        }
        self._encoder_mapping = {
            'dlsdk': EncoderDLSDKModel,
            'openvino': EncoderOVModel,
            'onnx_runtime': EncoderONNXModel,
            'dummy': DummyEncoder
        }
        self.num_processing_frames = network_info['decoder'].get('num_processing_frames', 16)
        self.processing_frames_buffer = []
        self.encoder = create_encoder(network_info['encoder'], launcher, self._encoder_mapping, delayed_model_loading)
        self.decoder = create_model(network_info['decoder'], launcher, self._decoder_mapping, 'decoder',
                                    delayed_model_loading)
        self.store_encoder_predictions = network_info['encoder'].get('store_predictions', False)
        self._encoder_predictions = [] if self.store_encoder_predictions else None
        self._part_by_name = {'encoder': self.encoder, 'decoder': self.decoder}
        self._raw_outs = OrderedDict()

    def predict(self, identifiers, input_data, encoder_callback=None):
        predictions, raw_outputs = [], []
        for data in input_data:
            encoder_prediction, decoder_inputs = self.encoder.predict(identifiers, data)
            if isinstance(encoder_prediction, tuple):
                encoder_prediction, raw_encoder_prediction = encoder_prediction
            else:
                raw_encoder_prediction = encoder_prediction
            if encoder_callback:
                encoder_callback(raw_encoder_prediction)
            if self.store_encoder_predictions:
                self._encoder_predictions.append(encoder_prediction)
            raw_output, prediction = self.decoder.predict(identifiers, decoder_inputs)
            raw_outputs.append(raw_output)
            predictions.append(prediction)
        return raw_outputs, predictions

    def reset(self):
        self.processing_frames_buffer = []
        if self._encoder_predictions is not None:
            self._encoder_predictions = []

    @property
    def adapter(self):
        return self.decoder.adapter

    def save_encoder_predictions(self):
        if self._encoder_predictions is not None:
            prediction_file = Path(self.network_info['encoder'].get('predictions', 'encoder_predictions.pickle'))
            with prediction_file.open('wb') as file:
                pickle.dump(self._encoder_predictions, file)

    def _add_raw_encoder_predictions(self, encoder_prediction):
        for key, output in encoder_prediction.items():
            if key not in self._raw_outs:
                self._raw_outs[key] = []
            self._raw_outs[key].append(output)


class EncoderDLSDKModel(BaseDLSDKModel):
    def predict(self, identifiers, input_data):
        input_data = self.fit_to_input(input_data)
        results = self.exec_network.infer(input_data)
        return results, results[self.output_blob]


class EncoderOVModel(BaseOpenVINOModel):
    def predict(self, identifiers, input_data):
        input_data = self.fit_to_input(input_data)
        results = self.infer(input_data, raw_results=True)
        return results, results[self.output_blob] if not isinstance(results, tuple) else results[0][self.output_blob]


class DecoderDLSDKModel(BaseDLSDKModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.adapter = create_adapter(network_info.get('adapter', 'ctc_greedy_decoder'))
        super().__init__(network_info, launcher, suffix, delayed_model_loading)
        self.adapter.output_blob = self.output_blob

    def predict(self, identifiers, input_data):
        feed_dict = self.fit_to_input(input_data)
        raw_result = self.exec_network.infer(feed_dict)
        result = self.adapter.process([raw_result], identifiers, [{}])

        return raw_result, result

    def set_input_and_output(self):
        super().set_input_and_output()
        self.adapter.output_blob = self.output_blob


class DecoderOVModel(BaseOpenVINOModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.adapter = create_adapter(network_info.get('adapter', 'ctc_greedy_decoder'))
        super().__init__(network_info, launcher, suffix, delayed_model_loading)
        self.adapter.output_blob = self.output_blob

    def predict(self, identifiers, input_data):
        feed_dict = self.fit_to_input(input_data)
        results, raw_results = self.infer(feed_dict, raw_results=True)
        result = self.adapter.process([results], identifiers, [{}])

        return raw_results, result

    def set_input_and_output(self):
        super().set_input_and_output()
        self.adapter.output_blob = self.output_blob


class EncoderONNXModel(BaseONNXModel):
    def predict(self, identifiers, input_data):
        results = self.inference_session.run((self.output_blob.name, ), self.fit_to_input(input_data))
        return results, results[0]


class DecoderONNXModel(BaseONNXModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        super().__init__(network_info, launcher, suffix, delayed_model_loading)
        self.adapter = create_adapter(network_info['adapter'])
        self.adapter.output_blob = self.output_blob.name

    def predict(self, identifiers, input_data):
        result = self.inference_session.run((self.output_blob.name,), self.fit_to_input(input_data))
        return result, self.adapter.process([{self.output_blob.name: result[0]}], identifiers, [{}])


class DummyEncoder:
    def __init__(self, network_info, launcher):
        self.network_info = network_info
        self.launcher = launcher
        if 'predictions' not in network_info:
            raise ConfigError('predictions_file is not found')
        self._predictions = read_pickle(network_info['predictions'])
        self.iterator = 0

    def predict(self, identifiers, input_data):
        result = self._predictions[self.iterator]
        self.iterator += 1
        return None, result

    def release(self):
        pass
