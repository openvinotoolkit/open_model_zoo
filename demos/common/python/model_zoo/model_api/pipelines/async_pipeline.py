"""
 Copyright (C) 2020-2023 Intel Corporation

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

from time import perf_counter
from typing import Dict, Set

from ..performance_metrics import PerformanceMetrics


def parse_devices(device_string):
    colon_position = device_string.find(':')
    if colon_position != -1:
        device_type = device_string[:colon_position]
        if device_type == 'HETERO' or device_type == 'MULTI':
            comma_separated_devices = device_string[colon_position + 1:]
            devices = comma_separated_devices.split(',')
            for device in devices:
                parenthesis_position = device.find(':')
                if parenthesis_position != -1:
                    device = device[:parenthesis_position]
            return devices
    return (device_string,)


def parse_value_per_device(devices: Set[str], values_string: str)-> Dict[str, int]:
    """Format: <device1>:<value1>,<device2>:<value2> or just <value>"""
    values_string_upper = values_string.upper()
    result = {}
    device_value_strings = values_string_upper.split(',')
    for device_value_string in device_value_strings:
        device_value_list = device_value_string.split(':')
        if len(device_value_list) == 2:
            if device_value_list[0] in devices:
                result[device_value_list[0]] = int(device_value_list[1])
        elif len(device_value_list) == 1 and device_value_list[0] != '':
            for device in devices:
                result[device] = int(device_value_list[0])
        elif device_value_list[0] != '':
            raise RuntimeError(f'Unknown string format: {values_string}')
    return result


def get_user_config(flags_d: str, flags_nstreams: str, flags_nthreads: int)-> Dict[str, str]:
    config = {}

    devices = set(parse_devices(flags_d))

    device_nstreams = parse_value_per_device(devices, flags_nstreams)
    for device in devices:
        if device == 'CPU':  # CPU supports a few special performance-oriented keys
            # limit threading for CPU portion of inference
            if flags_nthreads:
                config['CPU_THREADS_NUM'] = str(flags_nthreads)

            config['CPU_BIND_THREAD'] = 'NO'

            # for CPU execution, more throughput-oriented execution via streams
            config['CPU_THROUGHPUT_STREAMS'] = str(device_nstreams[device]) \
                if device in device_nstreams else 'CPU_THROUGHPUT_AUTO'
        elif device == 'GPU':
            config['GPU_THROUGHPUT_STREAMS'] = str(device_nstreams[device]) \
                if device in device_nstreams else 'GPU_THROUGHPUT_AUTO'
            if 'MULTI' in flags_d and 'CPU' in devices:
                # multi-device execution with the CPU + GPU performs best with GPU throttling hint,
                # which releases another CPU thread (that is otherwise used by the GPU driver for active polling)
                config['GPU_PLUGIN_THROTTLE'] = '1'
    return config


class AsyncPipeline:
    def __init__(self, model):
        self.model = model
        self.model.load()

        self.completed_results = {}
        self.callback_exceptions = []
        self.model.model_adapter.set_callback(self.callback)

        self.preprocess_metrics = PerformanceMetrics()
        self.inference_metrics = PerformanceMetrics()
        self.postprocess_metrics = PerformanceMetrics()

    def callback(self, request, callback_args):
        try:
            get_result_fn, (id, meta, preprocessing_meta, start_time) = callback_args
            self.completed_results[id] = (get_result_fn(request), meta, preprocessing_meta, start_time)
        except Exception as e:
            self.callback_exceptions.append(e)

    def submit_data(self, inputs, id, meta={}):
        preprocessing_start_time = perf_counter()
        inputs, preprocessing_meta = self.model.preprocess(inputs)
        self.preprocess_metrics.update(preprocessing_start_time)

        infer_start_time = perf_counter()
        callback_data = id, meta, preprocessing_meta, infer_start_time
        self.model.infer_async(inputs, callback_data)

    def get_raw_result(self, id):
        if id in self.completed_results:
            return self.completed_results.pop(id)
        return None

    def get_result(self, id):
        result = self.get_raw_result(id)
        if result:
            raw_result, meta, preprocess_meta, infer_start_time = result
            self.inference_metrics.update(infer_start_time)

            postprocessing_start_time = perf_counter()
            result = self.model.postprocess(raw_result, preprocess_meta), {**meta, **preprocess_meta}
            self.postprocess_metrics.update(postprocessing_start_time)
            return result
        return None

    def is_ready(self):
        return self.model.is_ready()

    def await_all(self):
        self.model.await_all()

    def await_any(self):
        self.model.await_any()
