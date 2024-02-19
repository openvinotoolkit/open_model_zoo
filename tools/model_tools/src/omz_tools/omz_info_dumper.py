# Copyright (c) 2019-2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import sys

from pathlib import Path

from omz_tools import _configuration, _common

def to_info(model):
    accuracy_config = _common.MODEL_ROOT / model.subdirectory / 'accuracy-check.yml'
    model_config = _common.MODEL_ROOT / model.subdirectory / 'model.yml'
    result = {
        'name': model.name,
        'composite_model_name': model.composite_model_name,

        'description': model.description,
        'framework': model.framework,
        'license_url': model.license_url,
        'accuracy_config': str(accuracy_config) if accuracy_config.exists() else None,
        'model_config': str(model_config) if model_config.exists() else None,
        'precisions': sorted(model.precisions),
        'subdirectory': str(model.subdirectory),
        'task_type': str(model.task_type),
        'input_info': [
            {'name': input.name, 'shape': input.shape, 'layout': input.layout} for input in model.input_info
        ],
        'model_stages': [],
    }

    for model_stage in model.model_stages:
        result['model_stages'].append(to_info(model_stage))
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', metavar='PAT[,PAT...]',
        help='only dump info for models whose names match at least one of the specified patterns')
    parser.add_argument('--list', type=Path, metavar='FILE.LST',
        help='only dump info for models whose names match at least one of the patterns in the specified file')
    parser.add_argument('--all', action='store_true', help='dump info for all available models')
    parser.add_argument('--print_all', action='store_true', help='print all available models')
    args = parser.parse_args()

    models = _configuration.load_models_from_args(parser, args, _common.MODEL_ROOT)

    json.dump(list(map(to_info, models)), sys.stdout, indent=4)
    print() # add a final newline

if __name__ == '__main__':
    main()
