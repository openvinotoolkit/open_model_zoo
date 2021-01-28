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

from collections import OrderedDict, namedtuple
from .base_profiler import PROFILERS_MAPPING, MetricProfiler, PROFILERS_WITH_DATA_IS_LIST

PorfilerID = namedtuple('ProfilerID', ['type', 'annotation_source', 'prediction_source', 'name'])


class ProfilingExecutor:
    def __init__(self, profile_report_type='csv'):
        self.profilers = OrderedDict()
        self._profiler_by_metric = OrderedDict()
        self.profile_report_type = profile_report_type

    def register_profiler_for_metric(self, metric_type, metric_name, annotation_source='', prediction_source=''):
        profiler = None
        for metric_types, profiler_type in PROFILERS_MAPPING.items():
            if metric_type in metric_types:
                if profiler_type in PROFILERS_WITH_DATA_IS_LIST:
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
