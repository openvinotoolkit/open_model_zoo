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

import copy
from functools import partial
import numpy as np

from .mtcnn_models import MTCNNCascadeModel
from .mtcnn_evaluator_utils import transform_for_callback
from .base_custom_evaluator import BaseCustomEvaluator


class MTCNNEvaluator(BaseCustomEvaluator):
    def __init__(self, dataset_config, launcher, model, orig_config):
        super().__init__(dataset_config, launcher, orig_config)
        self.model = model
        if hasattr(self.model, 'adapter') and self.model.adapter is not None:
            self.adapter_type = self.model.adapter.__provider__

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, _ = cls.get_dataset_and_launcher_info(config)
        model = MTCNNCascadeModel(
            config.get('network_info', {}), launcher, config.get('_models', []), delayed_model_loading
        )
        return cls(dataset_config, launcher, model, orig_config)

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
            for stage in self.model.stages.values():
                previous_stage_predictions = batch_prediction
                filled_inputs, batch_meta = stage.preprocess_data(
                    copy.deepcopy(batch_inputs), batch_annotation, previous_stage_predictions
                )
                batch_raw_prediction = stage.predict(filled_inputs, batch_meta, intermediate_callback)
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
                output_callback(transform_for_callback(batch_size, batch_raw_prediction),
                                metrics_result=metrics_result, element_identifiers=batch_identifiers,
                                dataset_indices=batch_input_ids)
            self._update_progress(progress_reporter, metric_config, batch_id, len(batch_prediction), csv_file)

    def reset(self):
        self.model.reset()

    def select_dataset(self, dataset_tag):
        super().select_dataset(dataset_tag)
        for _, stage in self.model.stages.items():
            stage.update_preprocessing(self.preprocessor)
