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
import collections
import json
import os
import shutil
import string
import sys

from openvino import Core, save_model
from pathlib import Path

from omz_tools import (
    _configuration, _common, _concurrency, _reporting,
)
from omz_tools.download_engine import validation

ModelOptimizerProperties = collections.namedtuple('ModelOptimizerProperties',
    ['cmd_prefix', 'extra_args', 'base_dir'])

def run_pre_convert(reporter, model, output_dir, args):
    script = _common.MODEL_ROOT / model.subdirectory / 'pre-convert.py'
    if not script.exists():
        return True

    reporter.print_section_heading('{}Running pre-convert script for {}',
        '(DRY RUN) ' if args.dry_run else '', model.name)

    cmd = [str(args.python), '--', str(script), '--',
        str(args.download_dir / model.subdirectory), str(output_dir / model.subdirectory)]

    reporter.print('Pre-convert command: {}', _common.command_string(cmd))
    reporter.print(flush=True)

    success = True if args.dry_run else reporter.job_context.subprocess(cmd)
    reporter.print()

    return success

def convert_to_onnx(reporter, model, output_dir, args, template_variables):
    reporter.print_section_heading('{}Converting {} to ONNX',
        '(DRY RUN) ' if args.dry_run else '', model.name)

    converter_path = Path(__file__).absolute().parent / \
        'internal_scripts' / model.converter_to_onnx

    conversion_to_onnx_args = [string.Template(arg).substitute(template_variables)
                               for arg in model.conversion_to_onnx_args]

    cmd = [str(args.python), '--', str(converter_path), *conversion_to_onnx_args]

    reporter.print('Conversion to ONNX command: {}', _common.command_string(cmd))
    reporter.print(flush=True)

    success = True if args.dry_run else reporter.job_context.subprocess(cmd)
    reporter.print()

    return success

def convert(reporter, model, output_dir, args, ovc_props, requested_precisions):
    telemetry = _common.Telemetry()
    if model.mo_args is None:
        reporter.print_section_heading('Skipping {} (no conversions defined)', model.name)
        reporter.print()
        return True

    model_precisions = requested_precisions & model.precisions
    if not model_precisions:
        reporter.print_section_heading('Skipping {} (all conversions skipped)', model.name)
        reporter.print()
        return True

    (output_dir / model.subdirectory).mkdir(parents=True, exist_ok=True)

    if not run_pre_convert(reporter, model, output_dir, args):
        telemetry.send_event('md', 'converter_failed_models', model.name)
        telemetry.send_event('md', 'converter_error',
            json.dumps({'error': 'pre-convert-script-failed', 'model': model.name, 'precision': None}))
        return False

    model_format = model.framework
    ovc_extension_dir = ovc_props.base_dir / 'extensions'
    if not ovc_extension_dir.exists():
        ovc_extension_dir = ovc_props.base_dir

    template_variables = {
        'config_dir': _common.MODEL_ROOT / model.subdirectory,
        'conv_dir': output_dir / model.subdirectory,
        'dl_dir': args.download_dir / model.subdirectory,
        'ovc_dir': ovc_props.base_dir,
        'ovc_ext_dir':  ovc_extension_dir,
    }

    if model.conversion_to_onnx_args:
        if not convert_to_onnx(reporter, model, output_dir, args, template_variables):
            telemetry.send_event('md', 'converter_failed_models', model.name)
            telemetry.send_event('md', 'converter_error',
                json.dumps({'error': 'convert_to_onnx-failed', 'model': model.name, 'precision': None}))
            return False
        model_format = 'onnx'

    custom_ovc_args = [
        string.Template(arg).substitute(template_variables)
        for arg in model.ovc_args]
    
    expanded_ovc_args = []
    input_model = ''
    for arg in custom_ovc_args:
        if 'input_model' in arg:
            input_model = arg.split('=')[-1]
            reporter.print(f'Model {input_model}')
        # if 'output' in arg:
        #     expanded_ovc_args.append(arg)

    for model_precision in sorted(model_precisions):
        data_type = model_precision.split('-')[0]
        shape_string = ','.join(str(input.shape).replace(" ", "") for input in model.input_info if input.shape)

        if shape_string:
            expanded_ovc_args.append('--input={}'.format(shape_string))
        if data_type == "FP16":
            expanded_ovc_args.append("--compress_to_fp16=True")
        else:
            expanded_ovc_args.append("--compress_to_fp16=False")

        ovc_output_dir = output_dir / model.subdirectory / model_precision
        ovc_cmd = [*ovc_props.cmd_prefix,
            input_model,
            f'--output_model={str(Path(ovc_output_dir, model.name + ".xml"))}',
            *expanded_ovc_args]

        reporter.print_section_heading('{}Converting {} to IR ({})',
            '(DRY RUN) ' if args.dry_run else '', model.name, model_precision)

        reporter.print('Conversion command: {}', _common.command_string(ovc_cmd))

        if not args.dry_run:
            reporter.print(flush=True)

            if not reporter.job_context.subprocess(ovc_cmd):
                telemetry.send_event('md', 'converter_failed_models', model.name)
                telemetry.send_event('md', 'converter_error',
                    json.dumps({'error': 'ovc-failed', 'model': model.name, 'precision': model_precision}))
                return False

        reporter.print()
        core = Core()
        core.set_property({"ENABLE_MMAP": False})
        rt_model = core.read_model(str(ovc_output_dir / model.name) + '.xml')
        try:
            val = validation.validate_string('model_type', model.model_info['model_type'])
            rt_model.set_rt_info(val, ['model_info', 'model_type'])
        except KeyError:
            pass
        try:
            val = validation.validate_nonnegative_float('confidence_threshold', model.model_info['confidence_threshold'])
            rt_model.set_rt_info(val, ['model_info', 'confidence_threshold'])
        except KeyError:
            pass
        try:
            val = validation.validate_nonnegative_float('iou_threshold', model.model_info['iou_threshold'])
            rt_model.set_rt_info(val, ['model_info', 'iou_threshold'])
        except KeyError:
            pass
        try:
            val = validation.validate_string('resize_type', model.model_info['resize_type'])
            rt_model.set_rt_info(val, ['model_info', 'resize_type'])
        except KeyError:
            pass
        try:
            val = validation.validate_list('anchors', model.model_info['anchors'])
            rt_model.set_rt_info(val, ['model_info', 'anchors'])
        except KeyError:
            pass
        try:
            val = validation.validate_list('masks', model.model_info['masks'])
            rt_model.set_rt_info(val, ['model_info', 'masks'])
        except KeyError:
            pass
        try:
            val = validation.validate_list('labels', model.model_info['labels'])
            rt_model.set_rt_info(val, ['model_info', 'labels'])
        except KeyError:
            pass
        save_model(rt_model, str(ovc_output_dir / model.name) + '.xml', "FP16" == data_type)
    return True

def num_jobs_arg(value_str):
    if value_str == 'auto':
        return os.cpu_count() or 1

    try:
        value = int(value_str)
        if value > 0: return value
    except ValueError:
        pass

    raise argparse.ArgumentTypeError('must be a positive integer or "auto" (got {!r})'.format(value_str))

def converter(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--download_dir', type=Path, metavar='DIR',
        default=Path.cwd(), help='root of the directory tree with downloaded model files')
    parser.add_argument('-o', '--output_dir', type=Path, metavar='DIR',
        help='root of the directory tree to place converted files into')
    parser.add_argument('--name', metavar='PAT[,PAT...]',
        help='convert only models whose names match at least one of the specified patterns')
    parser.add_argument('--list', type=Path, metavar='FILE.LST',
        help='convert only models whose names match at least one of the patterns in the specified file')
    parser.add_argument('--all', action='store_true', help='convert all available models')
    parser.add_argument('--print_all', action='store_true', help='print all available models')
    parser.add_argument('--precisions', metavar='PREC[,PREC...]',
        help='run only conversions that produce models with the specified precisions')
    parser.add_argument('-p', '--python', type=Path, metavar='PYTHON', default=sys.executable,
        help='Python executable to run OVC Converter with')
    parser.add_argument('--add_ovc_arg', dest='extra_ovc_args', metavar='ARG', action='append',
        help='Extra argument to pass to OVC Converter')
    parser.add_argument('--dry_run', action='store_true',
        help='Print the conversion commands without running them')
    parser.add_argument('-j', '--jobs', type=num_jobs_arg, default=1,
        help='number of conversions to run concurrently')

    # aliases for backwards compatibility
    parser.add_argument('--mo', type=Path, metavar='MO.PY',
        help='OVC Converter entry point script')
    parser.add_argument('--add_mo_arg', dest='extra_ovc_args', metavar='ARG', action='append',
        help='Extra argument to pass to OVC Converter')
    parser.add_argument('--add-mo-arg', dest='extra_ovc_args', action='append', help=argparse.SUPPRESS)
    parser.add_argument('--dry-run', action='store_true', help=argparse.SUPPRESS)

    args = parser.parse_args(argv)

    with _common.telemetry_session('Model Converter', 'converter') as telemetry:
        args_count = sum([args.all, args.name is not None, args.list is not None, args.print_all])
        if args_count == 0:
            telemetry.send_event('md', 'converter_selection_mode', None)
        else:
            for mode in ['all', 'list', 'name', 'print_all']:
                if getattr(args, mode):
                    telemetry.send_event('md', 'converter_selection_mode', mode)

        models = _configuration.load_models_from_args(parser, args, _common.MODEL_ROOT)

        if args.precisions is None:
            requested_precisions = _common.KNOWN_PRECISIONS
        else:
            requested_precisions = set(args.precisions.split(','))

        for model in models:
            precisions_to_send = requested_precisions if args.precisions else requested_precisions & model.precisions
            model_information = {
                'name': model.name,
                'framework': model.framework,
                'precisions': str(sorted(precisions_to_send)).replace(',', ';'),
            }
            telemetry.send_event('md', 'converter_model', json.dumps(model_information))

        unknown_precisions = requested_precisions - _common.KNOWN_PRECISIONS
        if unknown_precisions:
            sys.exit('Unknown precisions specified: {}.'.format(', '.join(sorted(unknown_precisions))))

        ovc_path = args.mo

        if ovc_path is None:
            ovc_executable = shutil.which('ovc')

            if ovc_executable:
                ovc_path = Path(ovc_executable)
            else:
                try:
                    # ovc_path = Path(os.environ['INTEL_OPENVINO_DIR']) / 'tools/mo/openvino/tools/mo/mo.py'
                    # if not ovc_path.exists():
                    ovc_path = Path(os.environ['INTEL_OPENVINO_DIR']) / 'tools/ovc/ovc.py'
                except KeyError:
                    sys.exit('Unable to locate OVC Converter. '
                        + 'Use --ovc or run setupvars.sh/setupvars.bat from the OpenVINO toolkit.')

        if ovc_path is not None:
            ovc_path = ovc_path.resolve()
            ovc_cmd_prefix = [str(args.python), '--', str(ovc_path)]

            if str(ovc_path).lower().endswith('.py'):
                ovc_dir = ovc_path.parent
            else:
                ovc_package_path, stderr = _common.get_package_path(args.python, 'openvino.tools.ovc')
                ovc_dir = ovc_package_path

                if ovc_package_path is None:
                    ovc_package_path, stderr = _common.get_package_path(args.python, 'ovc')
                    if ovc_package_path is None:
                        sys.exit('Unable to load Model Optimizer. Errors occurred: {}'.format(stderr))
                    ovc_dir = ovc_package_path.parent

        output_dir = args.download_dir if args.output_dir is None else args.output_dir

        reporter = _reporting.Reporter(_reporting.DirectOutputContext())
        ovc_props = ModelOptimizerProperties(
            cmd_prefix=ovc_cmd_prefix,
            extra_args=args.extra_ovc_args or [],
            base_dir=ovc_dir,
        )
        shared_convert_args = (output_dir, args, ovc_props, requested_precisions)

        def convert_model(model, reporter):
            if model.model_stages:
                results = []
                for model_stage in model.model_stages:
                    results.append(convert(reporter, model_stage, *shared_convert_args))
                return sum(results) == len(model.model_stages)
            else:
                return convert(reporter, model, *shared_convert_args)

        if args.jobs == 1 or args.dry_run:
            results = [convert_model(model, reporter) for model in models]
        else:
            results = _concurrency.run_in_parallel(args.jobs,
                lambda context, model:
                    convert_model(model, _reporting.Reporter(context)),
                models)

        failed_models = [model.name for model, successful in zip(models, results) if not successful]

        if failed_models:
            reporter.print('FAILED:')
            for failed_model_name in failed_models:
                reporter.print(failed_model_name)
            sys.exit(1)

def main():
    converter(sys.argv[1:])


if __name__ == '__main__':
    main()
