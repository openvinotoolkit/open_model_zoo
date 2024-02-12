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

import copy
from functools import partial
import numpy as np

from .mtcnn_models import build_stages
from .mtcnn_evaluator_utils import transform_for_callback
from .base_custom_evaluator import BaseCustomEvaluator
from ..quantization_model_evaluator import create_dataset_attributes


class MTCNNEvaluator(BaseCustomEvaluator):
    def __init__(self, dataset_config, launcher, stages, orig_config):
        super().__init__(dataset_config, launcher, orig_config)
        self.stages = stages
        stage = next(iter(self.stages.values()))
        if hasattr(stage, 'adapter') and stage.adapter is not None:
            self.adapter_type = stage.adapter.__provider__

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, _ = cls.get_dataset_and_launcher_info(config)
        models_info = config['network_info']
        stages = build_stages(models_info, [], launcher, config.get('_models'), delayed_model_loading)
        return cls(dataset_config, launcher, stages, orig_config)

    def _process(self, output_callback, calculate_metrics, progress_reporter, metric_config, csv_file):
        def no_detections(batch_pred):
            return batch_pred[0].size == 0

        for batch_id, (batch_input_ids, batch_annotation, batch_inputs, batch_identifiers) in enumerate(self.dataset):
            batch_prediction = []
            batch_raw_prediction = []
            intermediate_callback = None
            if output_callback:
                intermediate_callback = partial(output_callback, metrics_result=None,
                                                element_identifiers=batch_identifiers, dataset_indices=batch_input_ids)
            batch_size = 1
            for stage in self.stages.values():
                previous_stage_predictions = batch_prediction
                filled_inputs, batch_meta = stage.preprocess_data(
                    copy.deepcopy(batch_inputs), batch_annotation, previous_stage_predictions
                )
                batch_raw_prediction = stage.predict(filled_inputs, batch_meta, intermediate_callback)
                if isinstance(batch_raw_prediction, tuple):
                    batch_raw_prediction, _batch_raw_prediction = batch_raw_prediction
                else:
                    _batch_raw_prediction = batch_raw_prediction
                batch_size = np.shape(next(iter(filled_inputs[0].values())))[0]
                batch_prediction = stage.postprocess_result(
                    batch_identifiers, batch_raw_prediction, batch_meta, previous_stage_predictions
                )
                if no_detections(batch_prediction):
                    break
            batch_annotation, batch_prediction = self.postprocessor.process_batch(batch_annotation, batch_prediction)
            metrics_result = self._get_metrics_result(batch_input_ids, batch_annotation, batch_prediction,
                                                      calculate_metrics)
            if output_callback:
                output_callback(transform_for_callback(batch_size, _batch_raw_prediction),
                                metrics_result=metrics_result, element_identifiers=batch_identifiers,
                                dataset_indices=batch_input_ids)
            self._update_progress(progress_reporter, metric_config, batch_id, len(batch_prediction), csv_file)

    def _release_model(self):
        for _, stage in self.stages.items():
            stage.release()

    def reset(self):
        super().reset()
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
                self.stages[stage_name].load_network(network_, self.launcher, stage_name + '_')

    def load_network_from_ir(self, models_list):
        for models_dict in models_list:
            stage_name = models_dict['name']
            self.stages[stage_name].load_model(models_dict, self.launcher, stage_name + '_')

    def get_network(self):
        return [{'name': stage_name, 'model': stage.network} for stage_name, stage in self.stages.items()]

    def select_dataset(self, dataset_tag):
        if self.dataset is not None and isinstance(self.dataset_config, list):
            return
        dataset_attributes = create_dataset_attributes(self.dataset_config, dataset_tag)
        self.dataset, self.metric_executor, preprocessor, self.postprocessor = dataset_attributes
        for _, stage in self.stages.items():
            stage.update_preprocessing(preprocessor)
