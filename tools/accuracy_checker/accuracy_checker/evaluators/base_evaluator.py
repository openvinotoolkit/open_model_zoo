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

from ..presenters import write_csv_result

# pylint: disable=W0223
class BaseEvaluator:
    # create class instance using config
    @classmethod
    def from_configs(cls, config, *args, **kwargs):
        return cls()

    # extract information related to evaluation from config
    @staticmethod
    def get_processing_info(config):
        return config['name'], 'framework', 'device', None, 'dataset_name'

    # determine cycle for dataset processing
    def process_dataset(self, *args, **kwargs):
        raise NotImplementedError

    # finalize and get metrics results
    def compute_metrics(self, print_results=True, ignore_results_formatting=False, ignore_metric_reference=False):
        raise NotImplementedError

    # delayed metrics results logging
    def print_metrics_results(self, ignore_results_formatting=False, ignore_metric_reference=False):
        raise NotImplementedError

    # extract metrics results values prepared for printing
    def extract_metrics_results(self, print_results=True, ignore_results_formatting=False,
                                ignore_metric_reference=False):
        raise NotImplementedError

    # destruction for entity, which can not be deleted automatically
    def release(self):
        pass

    # reset progress for metrics calculation
    def reset(self):
        raise NotImplementedError

    # helper for sending evaluation info
    @staticmethod
    def send_processing_info(sender):
        return {}

    def set_launcher_property(self, property_dict):
        pass

    # helper for writing intermediate metric results to csv file
    def write_results_to_csv(self, csv_file, ignore_results_formatting, metric_interval):
        if csv_file:
            metrics_results, metrics_meta = self.extract_metrics_results(
                print_results=False, ignore_results_formatting=ignore_results_formatting
            )
            processing_info = self.get_processing_info(self.config)
            write_csv_result(
                csv_file, processing_info, metrics_results, metric_interval, metrics_meta
            )
