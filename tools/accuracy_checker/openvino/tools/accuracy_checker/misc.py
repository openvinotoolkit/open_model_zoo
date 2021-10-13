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

from .config import ConfigReader
from .evaluators import ModelEvaluator, ModuleEvaluator

EVALUATION_MODE = {
    'models': ModelEvaluator,
    'evaluations': ModuleEvaluator
}


def get_metric_references(config_path, definitions_path, data_source, annotations_dir, subset=None, additional_info=None):
    args = {'config': config_path, 'definitions': definitions_path, 'source': data_source, 'annotations': annotations_dir}
    config, mode = ConfigReader.merge(args)
    evaluator_class = EVALUATION_MODE.get(mode)
    if not evaluator_class:
        raise ValueError('Unknown evaluation mode')
    report = []
    for conf in config:
        report.extend(evaluator_class.provide_metric_references(conf, subset, additional_info))
    return report
