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

from argparse import Namespace
from pathlib import Path
from .config import ConfigReader
from .evaluators import ModelEvaluator, ModuleEvaluator
from .logging import init_logging

EVALUATION_MODE = {
    'models': ModelEvaluator,
    'evaluations': ModuleEvaluator
}


def get_metric_references(config_path, definitions_path=None, additional_info=None, return_header=True):
    init_logging()
    args = {'config': Path(config_path), 'definitions': Path(definitions_path) if definitions_path else None}
    if additional_info:
        args.update(additional_info)

    config, mode = ConfigReader.merge(Namespace(**args))
    evaluator_class = EVALUATION_MODE.get(mode)
    if not evaluator_class:
        raise ValueError('Unknown evaluation mode')
    report = []
    for conf in config[mode]:
        header, template_report = evaluator_class.provide_metric_references(conf)
        report.extend(template_report if isinstance(template_report, list) else [template_report])
    if return_header:
        return header, report
    return report
