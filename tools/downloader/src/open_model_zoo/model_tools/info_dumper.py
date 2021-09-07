# Copyright (c) 2019-2021 Intel Corporation
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

from open_model_zoo.model_tools import _configuration

def to_info(model):
    return {
        'name': model.name,
        'composite_model_name': model.composite_model_name,

        'description': model.description,
        'framework': model.framework,
        'license_url': model.license_url,
        'precisions': sorted(model.precisions),
        'quantization_output_precisions': sorted(model.quantization_output_precisions),
        'subdirectory': str(model.subdirectory),
        'task_type': str(model.task_type),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', metavar='PAT[,PAT...]',
        help='only dump info for models whose names match at least one of the specified patterns')
    parser.add_argument('--list', type=Path, metavar='FILE.LST',
        help='only dump info for models whose names match at least one of the patterns in the specified file')
    parser.add_argument('--all', action='store_true', help='dump info for all available models')
    parser.add_argument('--print_all', action='store_true', help='print all available models')
    args = parser.parse_args()

    models = _configuration.load_models_from_args(parser, args)

    json.dump(list(map(to_info, models)), sys.stdout, indent=4)
    print() # add a final newline

if __name__ == '__main__':
    main()
