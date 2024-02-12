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

import json
from collections import OrderedDict, namedtuple
from .base_profiler import PROFILERS_MAPPING, MetricProfiler, PROFILERS_WITH_DATA_IS_LIST

PorfilerID = namedtuple('ProfilerID', ['type', 'annotation_source', 'prediction_source', 'name'])


def write_summary_result(result, meta, out_path, label_map):
    summary = {}
    if isinstance(result, list):
        reversed_label_map = {v: k for k, v in label_map.items()} if label_map else {}
        mean_res, mean_meta, per_class_result = {}, {}, {}
        if meta[0].get('calculate_mean', True):
            for res, r_meta in zip(result, meta):
                if res['name'].endswith("@mean"):
                    mean_res = res
                    mean_meta = r_meta
                    continue
                class_name = r_meta.get('class_name', res['name'])
                class_id = reversed_label_map.get(class_name, class_name)
                per_class_result[str(class_id)] = {
                    'result': res['value'],
                    'result_scale': r_meta.get('scale', 100), 'result_postfix': r_meta.get('postfix', '%'),
                    'metric_target': r_meta['target']
                }
            summary['per_class_result'] = per_class_result
            if not mean_res:
                mean_res, mean_meta = result[0], meta[0]
        else:
            mean_res, mean_meta = result[0], meta[0]
    else:
        mean_res, mean_meta = result, meta

    summary['summary_result'] = {
        'metric_name': mean_res['name'], 'result': mean_res['value'], 'metric_type': mean_res['type'],
        'result_scale': mean_meta.get('scale', 100), 'result_postfix': mean_meta.get('postfix', '%'),
        'metric_target': mean_meta['target']
    }
    out_dict = {}
    if out_path.exists():
        with open(str(out_path), 'r', encoding='utf-8') as f:
            out_dict = json.load(f)

    final_summary = out_dict.get('summary_result', {})
    final_summary.update(summary['summary_result'])
    out_dict['summary_result'] = final_summary
    if 'per_class_result' in summary:
        per_class_res = out_dict.get('per_class_result', {})
        per_class_res.update(summary['per_class_result'])
        out_dict['per_class_result'] = per_class_res

    with open(str(out_path), 'w', encoding='utf-8') as f:
        json.dump(out_dict, f)


class ProfilingExecutor:
    def __init__(self, profile_report_type='csv'):
        self.profilers = OrderedDict()
        self._profiler_by_metric = OrderedDict()
        self.profile_report_type = profile_report_type

    def register_profiler_for_metric(self, metric_type, metric_name, annotation_source='', prediction_source=''):
        profiler = None
        for metric_types, profiler_type in PROFILERS_MAPPING.items():
            if metric_type in metric_types:
                if profiler_type in PROFILERS_WITH_DATA_IS_LIST or self.profile_report_type == 'json':
                    profiler_id = PorfilerID(profiler_type, annotation_source, prediction_source, metric_name)
                else:
                    profiler_id = PorfilerID(profiler_type, annotation_source, prediction_source, None)
                if profiler_id not in self.profilers:
                    self.profilers[profiler_id] = MetricProfiler.provide(
                        profiler_id.type, report_type=self.profile_report_type, name=profiler_id.name
                    )
                    self.profilers[profiler_id].set_dataset_meta(self.dataset_meta)
                self.profilers[profiler_id].register_metric(metric_name)
                self._profiler_by_metric[metric_name] = self.profilers[profiler_id]
                return self.profilers[profiler_id]

        return profiler

    def set_profiling_dir(self, profiler_dir):
        for profiler in self.profilers.values():
            profiler.set_output_dir(profiler_dir)

    def set_executing_info(self, processing_info):
        for profiler in self.profilers.values():
            profiler.set_processing_info(processing_info)

    def set_dataset_meta(self, meta):
        self.dataset_meta = meta

    def get_last_report(self):
        reports = OrderedDict()
        for profiler_id, profiler in self.profilers.items():
            reports[profiler_id] = profiler.last_report
        return reports

    def update_annotation_and_prediction(self, annotation, prediction):
        for profiler in self.profilers.values():
            if profiler.required_postprocessing:
                profiler.update_annotation_and_prediction(annotation, prediction)

    @property
    def required_postprocessing(self):
        for profiler in self.profilers.values():
            if profiler.required_postprocessing:
                return True
        return False
