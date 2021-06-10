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

from contextlib import contextmanager
import sys
import importlib
from pathlib import Path
from .base_evaluator import BaseEvaluator

# pylint:disable=R0904
class ModuleEvaluator(BaseEvaluator):
    def __init__(self, internal_module, config):
        super().__init__()
        self._internal_module = internal_module
        self._config = config

    @classmethod
    def from_configs(cls, config, *args, **kwargs):
        module = config['module']
        module_config = config.get('module_config')
        python_path = config.get('python_path')

        return cls(load_module(module, python_path).from_configs(module_config, *args, **kwargs), config)

    def process_dataset(self, stored_predictions, progress_reporter, *args, **kwargs):
        self._internal_module.process_dataset(
            *args, stored_predictions=stored_predictions, progress_reporter=progress_reporter, **kwargs
        )

    def compute_metrics(self, print_results=True, ignore_results_formatting=False):
        return self._internal_module.compute_metrics(print_results, ignore_results_formatting)

    def print_metrics_results(self, ignore_results_formatting=False):
        self._internal_module.print_metrics(ignore_results_formatting)

    def extract_metrics_results(self, print_results=True, ignore_results_formatting=False):
        return self._internal_module.extract_metrics_results(print_results, ignore_results_formatting)

    def release(self):
        self._internal_module.release()
        del self._internal_module

    def reset(self):
        self._internal_module.reset()

    def load_network(self, network=None):
        self._internal_module.load_network(network)

    def load_network_from_ir(self, models_dict):
        self._internal_module.load_network_from_ir(models_dict)

    def get_network(self):
        return self._internal_module.get_network()

    def get_metrics_attributes(self):
        return self._internal_module.get_metrics_attributes()

    def register_metric(self, metric_config):
        self._internal_module.register_metric(metric_config)

    def register_postprocessor(self, postprocessing_config):
        self._internal_module.register_postprocessor(postprocessing_config)

    def register_dumped_annotations(self):
        self._internal_module.register_dumped_annotations()

    def select_dataset(self, dataset_tag):
        self._internal_module.select_dataset(dataset_tag)

    def process_dataset_async(
            self,
            nreq=None,
            subset=None,
            num_images=None,
            check_progress=False,
            dataset_tag='',
            output_callback=None,
            allow_pairwise_subset=False,
            dump_prediction_to_annotation=False,
            **kwargs
    ):
        self._internal_module.process_dataset(
            subset=subset,
            num_images=num_images,
            check_progress=check_progress,
            dataset_tag=dataset_tag,
            output_callback=output_callback,
            allow_pairwise_subset=allow_pairwise_subset,
            dump_prediction_to_annotation=dump_prediction_to_annotation,
            **kwargs
        )

    @property
    def dataset(self):
        return self._internal_module.dataset

    @staticmethod
    def get_processing_info(config):
        module = config['module']
        python_path = config.get('python_path')
        return load_module(module, python_path).get_processing_info(config)

    def send_processing_info(self, sender):
        if sender is None:
            return {}
        return {'custom_evaluator': self._config['module']}

    def set_profiling_dir(self, profiler_dir):
        self._internal_module.set_profiling_dir(profiler_dir)

    @property
    def dataset_size(self):
        return self._internal_module.dataset_size


def load_module(model_cls, python_path=None):
    module_parts = model_cls.split(".")
    model_cls = module_parts[-1]
    module_as_path = '/'.join(module_parts[:-1]) + '.py'
    relative_path = Path(__file__).parent / module_as_path
    if not relative_path.exists():
        model_path = ".".join(module_parts[:-1])
        with append_to_path(python_path):
            module_cls = importlib.import_module(model_path).__getattribute__(model_cls)
            return module_cls
    model_path = ".{}".format(".".join(module_parts[:-1]))
    with append_to_path(python_path):
        package = ".".join(__name__.split(".")[:-1])
        module_cls = importlib.import_module(model_path, package=package).__getattribute__(model_cls)
        return module_cls


@contextmanager
def append_to_path(path):
    if path:
        sys.path.append(path)
    yield

    if path:
        sys.path.remove(path)
