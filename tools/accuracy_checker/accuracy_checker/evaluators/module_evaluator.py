"""
Copyright (c) 2019 Intel Corporation

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

from .base_evaluator import BaseEvaluator


class ModuleEvaluator(BaseEvaluator):
    def __init__(self, internal_module):
        super().__init__()
        self._internal_module = internal_module

    @classmethod
    def from_configs(cls, config):
        module = config['module']
        module_config = config.get('module_config')
        python_path = config.get('python_path')

        return cls(load_module(module, module_config, python_path))

    def process_dataset(self, stored_predictions, progress_reporter, *args, **kwargs):
        self._internal_module.process_dataset(stored_predictions, progress_reporter, *args, **kwargs)

    def compute_metrics(self, print_results=True, output_callback=None, ignore_results_formatting=False):
        self._internal_module.compute_metrics(print_results, output_callback, ignore_results_formatting)

    def release(self):
        self._internal_module.release()
        del self._internal_module

    def reset(self):
        self._internal_module.reset()

    def get_processing_info(self, config):
        return self._internal_module.get_processing_info(config)


def load_module(model_cls, module_config, python_path=None):
    module_parts = model_cls.split(".")
    model_cls = module_parts[-1]
    model_path = ".".join(module_parts[:-1])
    with append_to_path(python_path):
        model_cls = importlib.import_module(model_path).__getattribute__(model_cls)
        module = model_cls.from_configs(module_config)

        return module


@contextmanager
def append_to_path(path):
    if path:
        sys.path.append(path)
    yield

    if path:
        sys.path.remove(path)
