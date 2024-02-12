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
import numpy as np

from .base_custom_evaluator import BaseCustomEvaluator
from .base_models import (
    BaseDLSDKModel, BaseCascadeModel, BaseONNXModel, BaseOpenCVModel, BaseOpenVINOModel,
    create_model, create_encoder
)
from ...adapters import create_adapter
from ...config import ConfigError
from ...utils import contains_all, extract_image_representations, read_pickle, parse_partial_shape


class SequentialActionRecognitionEvaluator(BaseCustomEvaluator):
    def __init__(self, dataset_config, launcher, model, orig_config):
        super().__init__(dataset_config, launcher, orig_config)
        self.model = model
        if hasattr(self.model.decoder, 'adapter'):
            self.adapter_type = self.model.decoder.adapter.__provider__

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

        if self.model.store_encoder_predictions:
            self.model.save_encoder_predictions()


class SequentialModel(BaseCascadeModel):
    def __init__(self, network_info, launcher, models_args, is_blob, delayed_model_loading=False):
        super().__init__(network_info, launcher)
        parts = ['encoder', 'decoder']
        network_info = self.fill_part_with_model(network_info, parts, models_args, is_blob, delayed_model_loading)
        if not contains_all(network_info, parts) and not delayed_model_loading:
            raise ConfigError('network_info should contain encoder and decoder fields')
        self.num_processing_frames = network_info['decoder'].get('num_processing_frames', 16)
        self.processing_frames_buffer = []
        self._encoder_mapping = {
            'dlsdk': EncoderDLSDKModel,
            'openvino': EncoderOpenVINO,
            'onnx_runtime': EncoderONNXModel,
            'opencv': EncoderOpenCVModel,
            'dummy': DummyEncoder
        }
        self._decoder_mapping = {
            'dlsdk': DecoderDLSDKModel,
            'openvino': DecoderOpenVINOModel,
            'onnx_runtime': DecoderONNXModel,
            'opencv': DecoderOpenCVModel
        }
        self.encoder = create_encoder(network_info['encoder'], launcher, self._encoder_mapping, delayed_model_loading)
        self.decoder = create_model(network_info['decoder'], launcher, self._decoder_mapping, 'decoder',
                                    delayed_model_loading)
        self.store_encoder_predictions = network_info['encoder'].get('store_predictions', False)
        self._encoder_predictions = [] if self.store_encoder_predictions else None
        self._part_by_name = {'encoder': self.encoder, 'decoder': self.decoder}

    def predict(self, identifiers, input_data, encoder_callback=None):
        raw_outputs = []
        predictions = []
        if len(np.shape(input_data)) == 5:
            input_data = input_data[0]
        for data in input_data:
            encoder_prediction = self.encoder.predict(identifiers, [data])
            if isinstance(encoder_prediction, tuple):
                encoder_prediction, raw_encoder_prediction = encoder_prediction
            else:
                raw_encoder_prediction = encoder_prediction
            if encoder_callback:
                encoder_callback(raw_encoder_prediction)
            self.processing_frames_buffer.append(encoder_prediction[self.encoder.output_blob])
            if self.store_encoder_predictions:
                self._encoder_predictions.append(encoder_prediction[self.encoder.output_blob])
            if len(self.processing_frames_buffer) == self.num_processing_frames:
                raw_output, prediction = self.decoder.predict(identifiers, [self.processing_frames_buffer])
                raw_outputs.append(raw_output)
                predictions.append(prediction)
                self.processing_frames_buffer = []

        return raw_outputs, predictions

    def reset(self):
        self.processing_frames_buffer = []
        if self._encoder_predictions is not None:
            self._encoder_predictions = []

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
        input_dict = self.fit_to_input(input_data)
        if not self.is_dynamic and self.dynamic_inputs:
            self._reshape_input({key: data.shape for key, data in input_dict.items()})
        return self.exec_network.infer(input_dict)

    def fit_to_input(self, input_data):
        input_data = np.transpose(input_data, (0, 3, 1, 2))
        has_info = hasattr(self.exec_network, 'input_info')
        if has_info:
            input_info = self.exec_network.input_info[self.input_blob].input_data
        else:
            input_info = self.exec_network.inputs[self.input_blob]
        if (hasattr(input_info, 'is_dynamic') and not input_info.is_dynamic) or input_info.shape:
            input_data = input_data.reshape(input_info.shape)

        return {self.input_blob: np.array(input_data)}


class EncoderOpenVINO(BaseOpenVINOModel):
    def predict(self, identifiers, input_data):
        input_dict = self.fit_to_input(input_data)
        if not self.is_dynamic and self.dynamic_inputs:
            self._reshape_input({key: data.shape for key, data in input_dict.items()})
        return self.infer(input_dict, raw_results=True)

    def fit_to_input(self, input_data):
        input_data = np.transpose(input_data, (0, 3, 1, 2))
        input_info = self.inputs[self.input_blob]
        if not input_info.get_partial_shape().is_dynamic:
            input_data = input_data.reshape(parse_partial_shape(input_info.shape))

        return {self.input_blob: np.array(input_data)}


class DecoderDLSDKModel(BaseDLSDKModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.adapter = create_adapter(network_info.get('adapter', 'classification'))
        self.num_processing_frames = network_info.get('num_processing_frames', 16)
        super().__init__(network_info, launcher, suffix, delayed_model_loading)
        self.adapter.output_blob = self.output_blob

    def predict(self, identifiers, input_data):
        input_dict = self.fit_to_input(input_data)
        if not self.is_dynamic and self.dynamic_inputs:
            self._reshape_input({key: data.shape for key, data in input_dict.items()})
        raw_result = self.exec_network.infer(input_dict)
        result = self.adapter.process([raw_result], identifiers, [{}])

        return raw_result, result

    def fit_to_input(self, input_data):
        has_info = hasattr(self.exec_network, 'input_info')
        input_info = (
            self.exec_network.input_info[self.input_blob].input_data
            if has_info else self.exec_network.inputs[self.input_blob]
        )
        if not getattr(input_info, 'is_dynamic', False):
            input_data = np.reshape(input_data, input_info.shape)
        return {self.input_blob: np.array(input_data)}


class DecoderOpenVINOModel(BaseOpenVINOModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.adapter = create_adapter(network_info.get('adapter', 'classification'))
        self.num_processing_frames = network_info.get('num_processing_frames', 16)
        super().__init__(network_info, launcher, suffix, delayed_model_loading)
        self.adapter.output_blob = self.output_blob

    def predict(self, identifiers, input_data):
        input_dict = self.fit_to_input(input_data)
        if not self.is_dynamic and self.dynamic_inputs:
            self._reshape_input({key: data.shape for key, data in input_dict.items()})
        raw_result, raw_node_result = self.infer(input_dict, raw_results=True)
        result = self.adapter.process([raw_result], identifiers, [{}])

        return raw_node_result, result

    def fit_to_input(self, input_data):
        input_info = self.inputs[self.input_blob]
        if not input_info.get_partial_shape().is_dynamic:
            input_data = np.reshape(input_data, input_info.shape)
        return {self.input_blob: np.array(input_data)}


class EncoderONNXModel(BaseONNXModel):
    def predict(self, identifiers, input_data):
        return self.inference_session.run((self.output_blob.name, ), self.fit_to_input(input_data))[0]

    def fit_to_input(self, input_data):
        input_data = np.transpose(input_data, (0, 3, 1, 2))
        input_data = input_data.reshape(self.input_blob.shape)

        return {self.input_blob.name: input_data}


class DecoderONNXModel(BaseONNXModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        super().__init__(network_info, launcher, suffix, delayed_model_loading)
        self.adapter = create_adapter('classification')
        self.adapter.output_blob = self.output_blob.name
        self.num_processing_frames = network_info.get('num_processing_frames', 16)

    def predict(self, identifiers, input_data):
        result = self.inference_session.run((self.output_blob.name,), self.fit_to_input(input_data))
        return self.adapter.process([{self.output_blob.name: result[0]}], identifiers, [{}])

    def fit_to_input(self, input_data):
        input_data = np.reshape(input_data, self.input_blob.shape)
        return {self.input_blob.name: input_data}


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
        return result

    def release(self):
        pass


class EncoderOpenCVModel(BaseOpenCVModel):
    def predict(self, identifiers, input_data):
        self.network.setInput(self.fit_to_input(input_data)[self.input_blob], self.input_blob)
        return self.network.forward([self.output_blob])[0]

    def fit_to_input(self, input_data):
        input_data = np.transpose(input_data, (0, 3, 1, 2))
        input_data = input_data.reshape(self.input_shape)

        return {self.input_blob: input_data.astype(np.float32)}


class DecoderOpenCVModel(BaseOpenCVModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        super().__init__(network_info, launcher, suffix, delayed_model_loading)
        self.adapter = create_adapter('classification')
        self.adapter.output_blob = self.output_blob
        self.num_processing_frames = network_info.get('num_processing_frames', 16)

    def predict(self, identifiers, input_data):
        self.network.setInput(self.fit_to_input(input_data)[self.input_blob], self.input_blob)
        result = self.network.forward([self.output_blob])[0]
        return self.adapter.process([{self.output_blob.name: result}], identifiers, [{}])

    def fit_to_input(self, input_data):
        input_data = np.reshape(input_data, self.input_shape)
        return {self.input_blob: input_data.astype(np.float32)}
