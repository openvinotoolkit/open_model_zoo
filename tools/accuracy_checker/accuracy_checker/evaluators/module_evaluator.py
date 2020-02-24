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

        return cls(load_module(module, python_path).from_configs(module_config))

    def process_dataset(self, stored_predictions, progress_reporter, *args, **kwargs):
        self._internal_module.process_dataset(stored_predictions, progress_reporter, *args, **kwargs)

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

    @staticmethod
    def get_processing_info(config):
        module = config['module']
        python_path = config.get('python_path')
        return load_module(module, python_path).get_processing_info(config)


def load_module(model_cls, python_path=None):
    module_parts = model_cls.split(".")
    model_cls = module_parts[-1]
    model_path = ".".join(module_parts[:-1])
    with append_to_path(python_path):
        module_cls = importlib.import_module(model_path).__getattribute__(model_cls)
        return module_cls


@contextmanager
def append_to_path(path):
    if path:
        sys.path.append(path)
    yield

    if path:
        sys.path.remove(path)
