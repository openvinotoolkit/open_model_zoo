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
from .base_profiler import MetricProfiler


class SegmentationMetricProfiler(MetricProfiler):
    __provider__ = 'segmentation'
    fields = ['identifier']

    def __init__(self, dump_iterations=100, report_type='csv', name=None):
        self.updated_fields = False
        self.names = []
        self.metric_names = []
        self.mask_as_polygon = report_type == 'json'

        super().__init__(dump_iterations, report_type, name)
        self.required_postprocessing = True
        self.annotation, self.prediction = None, None

    def register_metric(self, metric_name):
        self.metric_names.append(metric_name)

    def generate_profiling_json(self, identifier, cm, metric_result,
                                prediction_mask_polygon, annotation_mask_polygon, per_class_result, ignore_label):
        report = {'identifier': identifier}
        class_result = {}
        if self.prediction is not None:
            prediction_mask_polygon = self.prediction.to_polygon()
        for label, polygons in prediction_mask_polygon.items():
            if label == ignore_label:
                continue
            class_result[int(label)] = {'prediction_mask': [polygon.tolist() for polygon in polygons]}
        if self.annotation:
            annotation_mask_polygon = self.annotation.to_polygon()
        for label, polygons in annotation_mask_polygon.items():
            if label == ignore_label:
                continue
            if int(label) not in class_result:
                class_result[int(label)] = {'prediction_mask': []}
            class_result[int(label)].update({'annotation_mask': [polygon.tolist() for polygon in polygons]})
        report['confusion_matrix'] = cm.tolist()
        report['result'] = np.mean(metric_result)
        if per_class_result:
            for label, metric in per_class_result.items():
                if int(label) not in class_result:
                    continue
                class_result[int(label)]['result'] = metric
        report['per_class_result'] = class_result

        return report

    def generate_profiling_data(self, identifier, metric_name, cm, metric_result, predicted_mask,
                                prediction_mask_polygon, annotation_mask_polygon, per_class_result, ignore_label):
        if self.report_type == 'json':
            return self.generate_profiling_json(identifier, cm, metric_result,
                                prediction_mask_polygon, annotation_mask_polygon, per_class_result, ignore_label)
        dumping_dir = self.out_dir / 'dumped'
        if not dumping_dir.exists():
            dumping_dir.mkdir(parents=True)
        if self._last_profile and self._last_profile['identifier'] == identifier:
            report = self._last_profile
        else:
            dumped_file_name = identifier.split('.')[0] + '.npy'
            mask_file = dumping_dir / dumped_file_name
            if not mask_file.parent.exists():
                mask_file.parent.mkdir(parents=True)
            predicted_mask.dump(str(mask_file))
            if not self.updated_fields:
                self._create_fields(metric_result)
            report = {'identifier': identifier, 'predicted_mask': str(dumping_dir / dumped_file_name)}
        if np.isscalar(metric_result) or np.size(metric_result) == 1:
            report['{}_result'.format(metric_name)] = np.mean(metric_result)
            return report
        if not self.names:
            metrics_results = {
                'class {} ({})'.format(class_id, metric_name):
                result for class_id, result in enumerate(metric_result)
            }
        else:
            metrics_results = {}
            for name, result in zip(self.names, metric_result):
                metrics_results['{} ({})'.format(name, metric_name)] = result
        report.update(metrics_results)
        return report

    def _create_fields(self, metric_result):
        self.fields = ['identifier', 'predicted_mask']
        for metric_name in self.metric_names:
            if np.isscalar(metric_result) or np.size(metric_result) == 1:
                self.fields.append('result')
            else:
                if self.names:
                    self.fields.extend(['{} ({})'.format(name, metric_name) for name in self.names])
                else:
                    self.fields.extend(
                        ['class {} ({})'.format(class_id, metric_name) for class_id, _ in enumerate(metric_result)]
                    )
        if self.report_type == 'json':
            self.fields.append('confusion_matrix')
        self.updated_fields = True

    def set_dataset_meta(self, meta):
        self.dataset_meta = meta

    def update_annotation_and_prediction(self, annotation, prediction):
        self.annotation = annotation
        self.prediction = prediction
