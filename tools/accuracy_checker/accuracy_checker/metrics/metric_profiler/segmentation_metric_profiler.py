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

import numpy as np
from .base_profiler import MetricProfiler


class SegmentationMetricProfiler(MetricProfiler):
    __provider__ = 'segmentation'
    fields = ['identifier']

    def __init__(self, dump_iterations=100, report_type='csv', name=None):
        self.updated_fields = False
        self.names = []
        self.metric_names = []

        super().__init__(dump_iterations, report_type, name)

    def register_metric(self, metric_name):
        self.metric_names.append(metric_name)

    def generate_profiling_data(self, identifier, metric_name, cm, metric_result, predicted_mask):
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
            if self.report_type == 'json':
                report['confusion_matrix'] = cm.tolist()
        if np.isscalar(metric_result) or np.size(metric_result) == 1:
            report['{}_result'.format(metric_name)] = np.mean(metric_result)
            return report
        if not self.names:
            metrics_results = {
                'class {} ({})'.format(class_id, metric_name): result for class_id, result in enumerate(metric_result)
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
