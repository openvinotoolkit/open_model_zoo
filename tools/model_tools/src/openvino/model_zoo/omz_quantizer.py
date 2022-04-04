# Copyright (c) 2020-2022 Intel Corporation
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

from openvino.model_zoo import (
    _configuration, _common, _reporting,
)

from openvino.model_zoo.quantize_engine.quantizer import Quantizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=Path, metavar='DIR',
        default=Path.cwd(), help='root of the directory tree with the full precision model files')
    parser.add_argument('--dataset_dir', type=Path, help='root of the dataset directory tree')
    parser.add_argument('-o', '--output_dir', type=Path, metavar='DIR',
        help='root of the directory tree to place quantized model files into')
    parser.add_argument('--name', metavar='PAT[,PAT...]',
        help='quantize only models whose names match at least one of the specified patterns')
    parser.add_argument('--list', type=Path, metavar='FILE.LST',
        help='quantize only models whose names match at least one of the patterns in the specified file')
    parser.add_argument('--all', action='store_true', help='quantize all available models')
    parser.add_argument('--print_all', action='store_true', help='print all available models')
    parser.add_argument('-p', '--python', type=Path, metavar='PYTHON', default=sys.executable,
        help='Python executable to run Post-Training Optimization Toolkit with')
    parser.add_argument('--pot', type=Path, help='Post-Training Optimization Toolkit entry point script')
    parser.add_argument('--dry_run', action='store_true',
        help='print the quantization commands without running them')
    parser.add_argument('--precisions', metavar='PREC[,PREC...]',
        help='quantize only to the specified precisions')
    parser.add_argument('--target_device', help='target device for the quantized model')
    args = parser.parse_args()

    with _common.telemetry_session('Model Quantizer', 'quantizer') as telemetry:
        models = _configuration.load_models_from_args(parser, args, _common.MODEL_ROOT)

        for mode in ['all', 'list', 'name']:
            if getattr(args, mode):
                telemetry.send_event('md', 'quantizer_selection_mode', mode)

        if args.precisions is None:
            requested_precisions = _common.KNOWN_QUANTIZED_PRECISIONS.keys()
        else:
            requested_precisions = set(args.precisions.split(','))

        for model in models:
            model_information = {
                'name': model.name,
                'framework': model.framework,
                'precisions': str(requested_precisions).replace(',', ';'),
            }
            telemetry.send_event('md', 'quantizer_model', json.dumps(model_information))

        quantizer = Quantizer(args.python, requested_precisions, args.output_dir, args.model_dir, args.pot,
                              args.dataset_dir, args.dry_run)
        reporter = _reporting.Reporter(_reporting.DirectOutputContext())

        failed_models = quantizer.bulk_quantize(reporter, models, args.target_device)

        if failed_models:
            reporter.print('FAILED:')
            for failed_model_name in failed_models:
                reporter.print(failed_model_name)
            sys.exit(1)


if __name__ == '__main__':
    main()
