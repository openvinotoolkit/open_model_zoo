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
import numpy as np

from .base_custom_evaluator import BaseCustomEvaluator
from .base_models import (
    BaseDLSDKModel, BaseCascadeModel, BaseOpenVINOModel,
    create_model, create_encoder
)
from ...adapters import create_adapter
from ...config import ConfigError
from ...utils import contains_all, extract_image_representations, parse_partial_shape
from ...dataset import DataProvider

class MultiviewDataProvider(DataProvider):
    def __init__(self,
    data_reader, annotation_provider=None, tag='', dataset_config=None, data_list=None, subset=None,
    batch=None, subdirs=None
    ):
        super().__init__(data_reader, annotation_provider, tag, dataset_config, data_list, subset, batch)
        self.subdirs = subdirs

    def __getitem__(self, item):
        if self.batch is None or self._batch <= 0:
            self.batch = 1
        if self.size <= item * self.batch:
            raise IndexError
        batch_annotation = []
        batch_start = item * self.batch
        batch_end = min(self.size, batch_start + self.batch)
        batch_input_ids = self.subset[batch_start:batch_end] if self.subset else range(batch_start, batch_end)
        batch_identifiers = [self._data_list[idx] for idx in batch_input_ids]
        batch_input = [self.read_data(identifier=identifier) for identifier in batch_identifiers]
        if self.annotation_provider:
            batch_annotation = [self.annotation_provider[idx] for idx in batch_identifiers]

        return batch_input_ids, batch_annotation, batch_input, batch_identifiers

    def read_data(self, identifier):
        multi_idx = [f'{subdir}/{identifier}' for subdir in self.subdirs]
        data = self.data_reader(identifier=multi_idx)
        data.identfier = multi_idx
        return data


class MultiViewActionRecognitionEvaluator(BaseCustomEvaluator):
    def __init__(self, dataset_config, launcher, model, orig_config, view_subdirs=None):
        super().__init__(dataset_config, launcher, orig_config)
        self.model = model
        if hasattr(self.model.decoder, 'adapter'):
            self.adapter_type = self.model.decoder.adapter.__provider__
        self.view_subdirs = view_subdirs

    def select_dataset(self, dataset_tag):
        super().select_dataset(dataset_tag)
        self.dataset = MultiviewDataProvider(
            self.dataset.data_reader,
            self.dataset.annotation_provider,
            self.dataset.tag,
            self.dataset.dataset_config,
            batch=self.dataset.batch,
            subset=self.dataset.subset,
            subdirs=self.view_subdirs)

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, _ = cls.get_dataset_and_launcher_info(config)
        model = SequentialModel(
            config.get('network_info', {}), launcher, config.get('_models', []), config.get('_model_is_blob'),
            delayed_model_loading
        )
        view_subdirs = config.get('view_subdirs', [])
        return cls(dataset_config, launcher, model, orig_config, view_subdirs)

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
        }
        self._decoder_mapping = {
            'dlsdk': DecoderDLSDKModel,
            'openvino': DecoderOpenVINOModel,
        }
        self.encoder = create_encoder(network_info['encoder'], launcher, self._encoder_mapping, delayed_model_loading)
        self.decoder = create_model(network_info['decoder'], launcher, self._decoder_mapping, 'decoder',
                                    delayed_model_loading)
        self._part_by_name = {'encoder': self.encoder, 'decoder': self.decoder}

    def predict(self, identifiers, input_data, encoder_callback=None):
        raw_outputs = []
        predictions = []
        if len(np.shape(input_data)) == 5:
            input_data = input_data[0]
        encoder_preds = []
        for data in input_data:
            encoder_prediction = self.encoder.predict(identifiers, [data])
            if isinstance(encoder_prediction, tuple):
                encoder_prediction, raw_encoder_prediction = encoder_prediction
            else:
                raw_encoder_prediction = encoder_prediction
            if encoder_callback:
                encoder_callback(raw_encoder_prediction)
            encoder_preds.append(encoder_prediction[self.encoder.output_blob])
        raw_output, prediction = self.decoder.predict(identifiers, encoder_preds)
        raw_outputs.append(raw_output)
        predictions.append(prediction)

        return raw_outputs, predictions


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
        inputs = {}
        input_info = (
            self.exec_network.input_info
            if has_info else self.exec_network.inputs
        )
        for input_name, data in zip(input_info, input_data):
            info = input_info[input_name] if not has_info else input_info[input_name].input_data
            if not info.is_dynamic:
                data = np.reshape(data, input_info.shape)
                inputs[input_name] = data
        return inputs


class DecoderOpenVINOModel(BaseOpenVINOModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.adapter = create_adapter(network_info.get('adapter', 'classification'))
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
        inputs = {}
        for (input_name, input_info), data in zip(self.inputs.items(), input_data):
            if not input_info.get_partial_shape().is_dynamic:
                data = np.reshape(data, input_info.shape)
                inputs[input_name] = data
        return inputs
