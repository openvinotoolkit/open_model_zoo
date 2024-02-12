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

from .base_custom_evaluator import BaseCustomEvaluator
from .base_models import BaseCascadeModel, BaseDLSDKModel, BaseOpenVINOModel, create_model
from ...adapters import create_adapter
from ...config import ConfigError
from ...launcher.input_feeder import InputFeeder
from ...utils import contains_all, extract_image_representations, postprocess_output_name


class SequentialBackgroundMatting(BaseCustomEvaluator):
    def __init__(self, dataset_config, launcher, model, adapter, orig_config):
        super().__init__(dataset_config, launcher, orig_config)
        self.model = model
        self.adapter = adapter

    def _process(self, output_callback, calculate_metrics, progress_reporter, metric_config, csv_file):
        previous_video_id = None
        for batch_id, (batch_input_ids, batch_annotation, batch_inputs, batch_identifiers) in enumerate(self.dataset):
            if previous_video_id is None or previous_video_id != batch_annotation[0].video_id:
                rnn_inputs = self.model.reset_rnn_inputs(len(batch_inputs))
            batch_inputs = self.preprocessor.process(batch_inputs, batch_annotation)
            batch_meta = extract_image_representations(batch_inputs, meta_only=True)
            filled_inputs = self.input_feeder.fill_inputs(batch_inputs)
            for i, filled_input in enumerate(filled_inputs):
                filled_input.update(rnn_inputs[i])
            batch_raw_results, batch_results = self.model.predict(batch_identifiers, filled_inputs)
            batch_predictions = self.adapter.process(batch_results, batch_identifiers, batch_meta)
            previous_video_id = batch_annotation[0].video_id
            rnn_inputs = self.model.set_rnn_inputs(batch_results)
            annotation, prediction = self.postprocessor.process_batch(batch_annotation, batch_predictions)
            metrics_result = self._get_metrics_result(batch_input_ids, annotation, prediction, calculate_metrics)
            if output_callback:
                output_callback(batch_raw_results[0], metrics_result=metrics_result,
                                element_identifiers=batch_identifiers, dataset_indices=batch_input_ids)
            self._update_progress(progress_reporter, metric_config, batch_id, len(prediction), csv_file)

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, _ = cls.get_dataset_and_launcher_info(config)
        model = SequentialBackgroundMattingModel(
            config.get('network_info', {}), launcher, config.get('_models', []), config.get('_model_is_blob'),
            delayed_model_loading
        )
        adapter = create_adapter(config['launchers'][0].get('adapter', 'background_matting_with_pha_and_fgr'))
        return cls(dataset_config, launcher, model, adapter, orig_config)

    @property
    def input_feeder(self):
        return self.model.model.input_feeder

    @property
    def inputs(self):
        return self.input_feeder.network_inputs


class SequentialBackgroundMattingModel(BaseCascadeModel):
    def __init__(self, network_info, launcher, models_args, is_blob, delayed_model_loading=False):
        super().__init__(network_info, launcher)
        parts = ['background_matting_model']
        network_info = self.fill_part_with_model(network_info, parts, models_args, is_blob, delayed_model_loading)
        if not contains_all(network_info, parts) and not delayed_model_loading:
            raise ConfigError('network_info should contain segnet_model field')
        self._model_mapping = {
            'dlsdk': DLSDKSequentialBackgroundMattingModel,
            'openvino': OpenVINOModelSequentialBackgroundMattingModel,
        }
        self.model = create_model(
            network_info['background_matting_model'],
            launcher,
            self._model_mapping,
            'background_matting_model',
            delayed_model_loading
        )
        self._part_by_name = {'background_matting_model': self.model}

    def predict(self, identifiers, input_data):
        batch_raw_results = []
        batch_results = []
        for identifier, data in zip(identifiers, input_data):
            results, raw_results = self.model.predict(identifier, data)
            batch_raw_results.append(raw_results)
            batch_results.append(results)
        return batch_raw_results, batch_results

    def reset_rnn_inputs(self, batch_size):
        output = []
        for _ in range(batch_size):
            zeros_lstm_inputs = {}
            for lstm_input_name in self.launcher.lstm_inputs.keys():
                if hasattr(self.model.network, 'input_info'):
                    shape = self.model.network.input_info[lstm_input_name].input_data.shape
                else:
                    shape = self.model.inputs[lstm_input_name].shape
                zeros_lstm_inputs[lstm_input_name] = np.zeros(shape, dtype=np.float32)
            output.append(zeros_lstm_inputs)
        return output

    def set_rnn_inputs(self, outputs):
        result = []
        for output in outputs:
            batch_rnn_inputs = {}
            for input_name, output_name in self.launcher.lstm_inputs.items():
                batch_rnn_inputs[input_name] = output[postprocess_output_name(output_name, output)]
            result.append(batch_rnn_inputs)
        return result


class DLSDKSequentialBackgroundMattingModel(BaseDLSDKModel):
    def predict(self, identifiers, input_data):
        outputs = self.exec_network.infer(input_data)
        if isinstance(outputs, tuple):
            outputs, raw_outputs = outputs
        else:
            raw_outputs = outputs
        return outputs, raw_outputs

    def input_shape(self, input_name):
        return self.launcher.inputs[input_name]

    def load_network(self, network, launcher):
        super().load_network(network, launcher)
        self.launcher = launcher
        self.launcher.network = self.network
        self.input_feeder = InputFeeder(self.launcher.config.get('inputs', []), self.launcher.inputs, self.input_shape,
                                        self.launcher.fit_to_input, self.launcher.default_layout)


class OpenVINOModelSequentialBackgroundMattingModel(BaseOpenVINOModel):
    def predict(self, identifiers, input_data):
        return self.infer(input_data, raw_results=True)

    def input_shape(self, input_name):
        return self.launcher.inputs[input_name]

    def load_network(self, network, launcher):
        super().load_network(network, launcher)
        self.launcher = launcher
        self.launcher.network = self.network
        self.input_feeder = InputFeeder(self.launcher.config.get('inputs', []), self.launcher.inputs, self.input_shape,
                                        self.launcher.fit_to_input, self.launcher.default_layout)
