"""
Copyright (c) 2018-2022 Intel Corporation

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
from ...utils import contains_all, extract_image_representations


class SequentialBackgroundMatting(BaseCustomEvaluator):
    def __init__(self, dataset_config, launcher, model, input_feeder, adapter, orig_config):
        super().__init__(dataset_config, launcher, orig_config)
        self.model = model
        self.input_feeder = input_feeder
        self.adapter = adapter
        self.inputs = input_feeder.network_inputs

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
            batch_raw_results = self.model.predict(batch_identifiers, filled_inputs)
            batch_predictions = self.adapter.process(batch_raw_results, batch_identifiers, batch_meta)
            previous_video_id = batch_annotation[0].video_id
            rnn_inputs = self.model.set_rnn_inputs(batch_raw_results)
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
        launcher.network = model.model.network
        launcher_config = config['launchers'][0]
        postpone_model_loading = False
        input_precision = launcher_config.get('_input_precision', [])
        input_layouts = launcher_config.get('_input_layout', '')
        input_feeder = InputFeeder(
            launcher.config.get('inputs', []), model.model.network.inputs, cls.input_shape, launcher.fit_to_input,
            launcher.default_layout, launcher_config['framework'] == 'dummy' or postpone_model_loading, input_precision,
            input_layouts
        )
        adapter = create_adapter(launcher_config.get('adapter', 'background_matting_with_pha_and_fgr'))
        return cls(dataset_config, launcher, model, input_feeder, adapter, orig_config)

    def input_shape(self, input_name):
        return self.inputs[input_name]


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

    def predict(self, batch_identifiers, batch_input_data):
        batch_raw_results = []
        for identifier, input_data in zip(batch_identifiers, batch_input_data):
            raw_results = self.model.predict(identifier, input_data)
            batch_raw_results.append(raw_results)
        return batch_raw_results

    def reset_rnn_inputs(self, batch_size):
        output = []
        for _ in range(batch_size):
            zeros_lstm_inputs = {}
            for lstm_input_name, _ in self.launcher._lstm_inputs.items():
                shape = self.model.network.input_info[lstm_input_name].input_data.shape
                zeros_lstm_inputs[lstm_input_name] = np.zeros(shape, dtype=np.float32)
            output.append(zeros_lstm_inputs)
        return output

    def set_rnn_inputs(self, outputs):
        result = []
        for output in outputs:
            batch_rnn_inputs = {}
            for input_name, output_name in self.launcher._lstm_inputs.items():
                batch_rnn_inputs[input_name] = output[output_name]
            result.append(batch_rnn_inputs)
        return result


class DLSDKSequentialBackgroundMattingModel(BaseDLSDKModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        super().__init__(network_info, launcher, suffix, delayed_model_loading)

    def predict(self, identifiers, input_data):
        return self.exec_network.infer(input_data)


class OpenVINOModelSequentialBackgroundMattingModel(BaseOpenVINOModel):
    def __init__(self, network_info, launcher, suffix=None, delayed_model_loading=False):
        super().__init__(network_info, launcher, suffix, delayed_model_loading)

    def predict(self, identifiers, input_data):
        return self.exec_network.infer(input_data)
