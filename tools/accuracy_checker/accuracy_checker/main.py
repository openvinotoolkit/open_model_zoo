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

import json
import sys
from datetime import datetime
import cv2

from .argparser import build_arguments_parser
from .config import ConfigReader
from .logging import print_info, add_file_handler, exception, init_logging
from .evaluators import ModelEvaluator, ModuleEvaluator
from .progress_reporters import ProgressReporter
from .presenters import write_csv_result
from .utils import (
    validate_print_interval,
    start_telemetry,
    end_telemetry,
    send_telemetry_event
)

EVALUATION_MODE = {
    'models': ModelEvaluator,
    'evaluations': ModuleEvaluator
}


def main():
    init_logging()
    return_code = 0
    args = build_arguments_parser().parse_args()
    tm = start_telemetry()
    progress_bar_provider = args.progress if ':' not in args.progress else args.progress.split(':')[0]
    progress_reporter = ProgressReporter.provide(progress_bar_provider, None, print_interval=args.progress_interval)
    if args.log_file:
        add_file_handler(args.log_file)
    evaluator_kwargs = configure_evaluator_kwargs(args)
    details = {
        'mode': "online" if not args.store_only else "offline",
        'metric_profiling': args.profile or False,
        'error': None
    }

    config, mode = ConfigReader.merge(args)
    evaluator_class = EVALUATION_MODE.get(mode)
    if not evaluator_class:
        send_telemetry_event(tm, 'error', 'Unknown evaluation mode')
        end_telemetry(tm)
        raise ValueError('Unknown evaluation mode')
    for config_entry in config[mode]:
        details.update({'status': 'started', "error": None})
        send_telemetry_event(tm, 'status', 'started')
        config_entry.update({
            '_store_only': args.store_only,
            '_stored_data': args.stored_predictions
        })
        try:
            processing_info = evaluator_class.get_processing_info(config_entry)
            print_processing_info(*processing_info)
            evaluator = evaluator_class.from_configs(config_entry)
            details.update(evaluator.send_processing_info(tm))
            metric_types = details.get('metrics', [])
            if args.profile:
                setup_profiling(args.profiler_logs_dir, evaluator)
            send_telemetry_event(tm, 'model_run', json.dumps(details))
            for metric in metric_types:
                send_telemetry_event(tm, 'metric_type', metric)
            evaluator.process_dataset(
                stored_predictions=args.stored_predictions, progress_reporter=progress_reporter, **evaluator_kwargs
            )
            if not args.store_only:
                metrics_results, metrics_meta = evaluator.extract_metrics_results(
                    print_results=True, ignore_results_formatting=args.ignore_result_formatting,
                    ignore_metric_reference=args.ignore_metric_reference
                )
                if args.csv_result:
                    write_csv_result(
                        args.csv_result, processing_info, metrics_results, evaluator.dataset_size, metrics_meta
                    )
            evaluator.release()
            details['status'] = 'finished'
            send_telemetry_event(tm, 'status', 'success')
            send_telemetry_event(tm, 'model_run', json.dumps(details))

        except Exception as e:  # pylint:disable=W0703
            details['status'] = 'error'
            details['error'] = str(type(e))
            send_telemetry_event(tm, 'status', 'failure')
            send_telemetry_event(tm, 'model_run', json.dumps(details))
            exception(e)
            return_code = 1
            continue
    end_telemetry(tm)
    sys.exit(return_code)


def print_processing_info(model, launcher, device, tags, dataset):
    print_info('Processing info:')
    print_info('model: {}'.format(model))
    print_info('launcher: {}'.format(launcher))
    if tags:
        print_info('launcher tags: {}'.format(' '.join(tags)))
    print_info('device: {}'.format(device.upper()))
    print_info('dataset: {}'.format(dataset))
    print_info('OpenCV version: {}'.format(cv2.__version__))


def setup_profiling(logs_dir, evaluator):
    _timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    profiler_dir = logs_dir / _timestamp
    print_info('Metric profiling activated. Profiler output will be stored in {}'.format(profiler_dir))
    evaluator.set_profiling_dir(profiler_dir)


def configure_evaluator_kwargs(args):
    evaluator_kwargs = {}
    if args.intermediate_metrics_results:
        validate_print_interval(args.metrics_interval)
        evaluator_kwargs['intermediate_metrics_results'] = args.intermediate_metrics_results
        evaluator_kwargs['metrics_interval'] = args.metrics_interval
        evaluator_kwargs['ignore_result_formatting'] = args.ignore_result_formatting
        evaluator_kwargs['csv_result'] = args.csv_result
        evaluator_kwargs['ignore_metric_reference'] = args.ignore_metric_reference
    evaluator_kwargs['store_only'] = args.store_only
    return evaluator_kwargs


if __name__ == '__main__':
    main()
