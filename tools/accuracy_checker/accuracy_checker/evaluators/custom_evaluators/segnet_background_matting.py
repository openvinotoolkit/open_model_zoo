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

import numpy as np
from .sr_evaluator import SuperResolutionFeedbackEvaluator
from .base_models import BaseCascadeModel, create_model, BaseDLSDKModel, BaseONNXModel, BaseOpenVINOModel
from ...adapters import create_adapter
from ...utils import contains_all, generate_layer_name, extract_image_representations, postprocess_output_name
from ...config import ConfigError


class FeedbackModel:
    def set_feedback(self, feedback):
        if np.ndim(feedback) == 2:
            feedback = np.expand_dims(feedback, -1)
        if np.shape(feedback)[0] == 1:
            feedback = np.transpose(feedback, (1, 2, 0))
        if feedback.max() > 1:
            feedback = feedback.astype(np.float32) / 255
        self.feedback = feedback
        self._feedback_shape = feedback.shape

    def reset_state(self):
        if self._feedback_shape is None:
            self.feedback = None
        else:
            self.feedback = np.zeros(self._feedback_shape)


class ONNXFeedbackModel(FeedbackModel, BaseONNXModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.feedback = None
        self._feedback_shape = None
        self.adapter = create_adapter(network_info.get('adapter', 'background_matting'))
        super().__init__(network_info, launcher, suffix, delayed_model_loading)

    def predict(self, identifiers, input_data):
        raw_results = self.inference_session.run((self.output_blob.name,), self.fit_to_input(input_data))
        results = self.adapter.process([{self.output_blob.name: raw_results[0]}], identifiers, [{}])

        return {self.output_blob: raw_results[0]}, results[0]

    def fit_to_input(self, input_data):
        if self.feedback is None:
            h, w = input_data.shape[:2]
            self.feedback = np.zeros((h, w, 1), dtype=np.float32)
        return {
            self.input_blob.name: np.expand_dims(
                np.transpose(np.concatenate([input_data, self.feedback], -1), (2, 0, 1)), 0
            ).astype(np.float32)
        }


class DLSDKFeedbackModel(FeedbackModel, BaseDLSDKModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.feedback = None
        self._feedback_shape = None
        self.adapter = create_adapter(network_info.get('adapter', 'background_matting'))
        super().__init__(network_info, launcher, suffix, delayed_model_loading)

    def predict(self, identifiers, input_data):
        data = self.fit_to_input(input_data)
        if not self.is_dynamic and self.dynamic_inputs:
            self._reshape_input({key: in_data.shape for key, in_data in data.items()})
        raw_result = self.exec_network.infer(data)
        result = self.adapter.process([raw_result], identifiers, [{}])
        return raw_result, result[0]

    def fit_to_input(self, input_data):
        if self.feedback is None:
            h, w = input_data.shape[:2]
            self.feedback = np.zeros((h, w, 1), dtype=np.float32)
        return {self.input_blob: np.expand_dims(
            np.transpose(np.concatenate([input_data, self.feedback], -1), (2, 0, 1)), 0
        )}

    def set_input_and_output(self):
        has_info = hasattr(self.exec_network, 'input_info')
        input_info = self.exec_network.input_info if has_info else self.exec_network.inputs
        input_blob = next(iter(input_info))
        with_prefix = input_blob.startswith(self.default_model_suffix + '_')
        if self.input_blob is None:
            self.input_blob = input_blob
            self.output_blob = next(iter(self.exec_network.outputs))
        if with_prefix != self.with_prefix:
            self.input_blob = generate_layer_name(self.input_blob, self.default_model_suffix, with_prefix)

        self.with_prefix = with_prefix

    def load_network(self, network, launcher):
        super().load_network(network, launcher)
        self.set_input_and_output()


class OpenVINOFeedbackModel(FeedbackModel, BaseOpenVINOModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        self.feedback = None
        self._feedback_shape = None
        self.adapter = create_adapter(network_info.get('adapter', 'background_matting'))
        super().__init__(network_info, launcher, suffix, delayed_model_loading)

    def predict(self, identifiers, input_data):
        data = self.fit_to_input(input_data)
        if not self.is_dynamic and self.dynamic_inputs:
            self._reshape_input({key: in_data.shape for key, in_data in data.items()})
        raw_result = self.infer(data, raw_results=True)
        if isinstance(raw_result, tuple):
            return raw_result[1], self.adapter.process([raw_result[0]], identifiers, [{}])[0]
        result = self.adapter.process([raw_result], identifiers, [{}])
        return raw_result, result[0]

    def fit_to_input(self, input_data):
        if self.feedback is None:
            h, w = input_data.shape[:2]
            self.feedback = np.zeros((h, w, 1), dtype=np.float32)
        return {self.input_blob: np.expand_dims(
            np.transpose(np.concatenate([input_data, self.feedback], -1), (2, 0, 1)), 0
        )}

    def set_input_and_output(self):
        input_blob = next(iter(self.inputs))
        with_prefix = input_blob.startswith(self.default_model_suffix + '_')
        if self.input_blob is None:
            self.input_blob = input_blob
            self.output_blob = next(iter(self.outputs))
        if with_prefix != self.with_prefix:
            self.input_blob = generate_layer_name(self.input_blob, self.default_model_suffix, with_prefix)
        self.output_blob = postprocess_output_name(
            self.output_blob, self.outputs, additional_mapping=self.additional_output_mapping, raise_error=False)
        if self.output_blob not in self.outputs:
            self.output_blob = postprocess_output_name(
            generate_layer_name(self.output_blob, self.default_model_suffix, with_prefix),
            self.outputs, additional_mapping=self.additional_output_mapping, raise_error=False)
        self.adapter.output_blob = self.output_blob
        self.with_prefix = with_prefix

    def load_network(self, network, launcher):
        super().load_network(network, launcher)
        self.set_input_and_output()


class VideoBackgroundMatting(SuperResolutionFeedbackEvaluator):
    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, _ = cls.get_dataset_and_launcher_info(config)
        model = SegnetModel(
            config.get('network_info', {}), launcher, config.get('_models', []), config.get('_model_is_blob'),
            delayed_model_loading
        )
        return cls(dataset_config, launcher, model, orig_config)

    def _process(self, output_callback, calculate_metrics, progress_reporter, metric_config, csv_file):
        previous_video_id = ''
        for batch_id, (batch_input_ids, batch_annotation, batch_inputs, batch_identifiers) in enumerate(self.dataset):
            if previous_video_id != batch_identifiers[0].video_id:
                self.model.reset()
            batch_inputs = self.preprocessor.process(batch_inputs, batch_annotation)
            batch_inputs_extr, _ = extract_image_representations(batch_inputs)
            batch_raw_prediction, batch_prediction = self.model.predict(
                batch_identifiers, batch_inputs_extr
            )
            self.model.set_feedback(batch_prediction[0].value)
            previous_video_id = batch_prediction[0].identifier.video_id
            annotation, prediction = self.postprocessor.process_batch(batch_annotation, batch_prediction)
            metrics_result = self._get_metrics_result(batch_input_ids, annotation, prediction, calculate_metrics)
            if output_callback:
                output_callback(batch_raw_prediction[0], metrics_result=metrics_result,
                                element_identifiers=batch_identifiers, dataset_indices=batch_input_ids)
            self._update_progress(progress_reporter, metric_config, batch_id, len(prediction), csv_file)


class SegnetModel(BaseCascadeModel):
    def __init__(self, network_info, launcher, models_args, is_blob, delayed_model_loading=False):
        super().__init__(network_info, launcher)
        parts = ['segnet_model']
        network_info = self.fill_part_with_model(network_info, parts, models_args, is_blob, delayed_model_loading)
        if not contains_all(network_info, parts) and not delayed_model_loading:
            raise ConfigError('network_info should contain segnet_model field')
        self._model_mapping = {
            'dlsdk': DLSDKFeedbackModel,
            'openvino': OpenVINOFeedbackModel,
            'onnx_runtime': ONNXFeedbackModel,
        }
        self.model = create_model(network_info['segnet_model'], launcher, self._model_mapping, 'segnet_model',
                                  delayed_model_loading)
        self._part_by_name = {'segnet_model': self.model}

    def predict(self, identifiers, input_data):
        predictions, raw_outputs = [], []
        for data in input_data:
            output, prediction = self.model.predict(identifiers, data)
            raw_outputs.append(output)
            predictions.append(prediction)
        return raw_outputs, predictions

    def reset(self):
        self.model.reset_state()

    def set_feedback(self, feedback):
        self.model.set_feedback(feedback)
