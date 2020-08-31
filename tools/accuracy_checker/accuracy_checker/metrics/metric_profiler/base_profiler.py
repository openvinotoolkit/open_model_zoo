"""
Copyright (c) 2018-2020 Intel Corporation

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

import json
from csv import DictWriter
from pathlib import Path
from ...dependency import ClassProvider

PROFILERS_MAPPING = {
    (
        'accuracy',
        'accuracy_per_class',
        'classification_f1-score'
    ): 'classification',
    ('character_recognition_accuracy', ): 'char_classification',
    ('clip_accuracy', ): 'clip_classification',
    (
        'metthews_correlation_coef',
        'multi_accuracy',
        'multi_recall',
        'multi_precision',
        'f1-score'
    ): 'binary_classification',
    (
        'mae',
        'mse',
        'rmse',
        'mae_on_interval',
        'mse_on_interval',
        'rmse_on_interval',
        'angle_error'
    ): 'regression',
    ('psnr', 'ssim'): 'complex_regression',
    ('normed_error', 'per_point_normed_error'): 'point_regression',
    ('segmentation_accuracy', 'mean_iou', 'mean_accuracy', 'frequency_weighted_accuracy'): 'segmentation',
    ('coco_precision', 'coco_recall', 'map', 'recall', 'miss_rate'): 'detection',
    ('coco_orig_segm_precision', ): 'instance_segmentation'
}


class MetricProfiler(ClassProvider):
    __provider_class__ = 'metric_profiler'
    fields = ['identifier']

    def __init__(self, dump_iterations=100, report_type='csv'):
        self.report_type = report_type
        self.report_file = '{}.{}'.format(self.__provider__, report_type)
        self.out_dir = Path()
        self.dump_iterations = dump_iterations
        self.storage = []
        self.write_result = self.write_csv_result if report_type == 'csv' else self.write_json_result
        self._last_profile = None

    def register_metric(self, metric_name):
        self.fields.append('{}_result'.format(metric_name))

    def generate_profiling_data(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        profiling_data = self.generate_profiling_data(*args, **kwargs)
        self._last_profile = profiling_data
        if isinstance(profiling_data, list):
            finished = True
            if finished:
                self.storage.extend(profiling_data)
        else:
            finished = True
            if finished:
                self.storage.append(profiling_data)

        if len(self.storage) % self.dump_iterations == 0 and finished:
            self.write_result()
            self.storage = []

    def finish(self):
        if self.storage:
            self.write_result()

    def reset(self):
        self.storage = []

    def write_csv_result(self):
        out_path = self.out_dir / self.report_file
        new_file = not out_path.exists()

        with open(str(out_path), 'a+', newline='') as f:
            writer = DictWriter(f, fieldnames=self.fields)
            if new_file:
                writer.writeheader()
            writer.writerows(self.storage)

    def write_json_result(self):
        out_path = self.out_dir / self.report_file
        new_file = not out_path.exists()
        if not new_file:
            with open(str(out_path), 'r') as f:
                out_dict = json.load(f)
            out_dict['report'].extend(self.storage)
        else:
            out_dict = {
                'processing_info': {
                    'model': self.model_name,
                    'dataset': self.dataset,
                    'framework': self.framework,
                    'device': self.device,
                    'tags': self.tags
                },
                'report': self.storage,
                'report_type': self.__provider__,
                'dataset_meta': self.dataset_meta
            }
        with open(str(out_path), 'w') as f:
            json.dump(out_dict, f)

    def set_output_dir(self, out_dir):
        self.out_dir = out_dir
        if not out_dir.exists():
            self.out_dir.mkdir(parents=True)

    def set_processing_info(self, processing_info):
        self.model_name, self.framework, self.device, self.tags, self.dataset = processing_info
        self.report_file = '{}_{}_{}_{}_{}_{}'.format(
            self.model_name, self.framework, self.device, '_'.join(self.tags or []), self.dataset, self.report_file
        )

    def set_dataset_meta(self, meta):
        self.dataset_meta = meta

    @property
    def last_report(self):
        return self._last_profile


def create_profiler(metric_type, metric_name):
    profiler = None
    for metric_types, profiler_id in PROFILERS_MAPPING.items():
        if metric_type in metric_types:
            return MetricProfiler.provide(profiler_id, metric_name)
    return profiler
