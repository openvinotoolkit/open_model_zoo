"""
 Copyright (C) 2020 Intel Corporation

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

import cv2
import logging as log

from pipelines import parse_devices

def put_highlighted_text(frame, message, position, font_face, font_scale, color, thickness):
    cv2.putText(frame, message, position, font_face, font_scale, (255, 255, 255), thickness + 1) # white border
    cv2.putText(frame, message, position, font_face, font_scale, color, thickness)

def resolution(value):
    try:
        result = [int(v) for v in value.split('x')]
        if len(result) != 2:
            raise RuntimeError('Сorrect format of --output_resolution parameter is "width"x"height".')
    except ValueError:
        raise RuntimeError('Сorrect format of --output_resolution parameter is "width"x"height".')
    return result

def log_blobs_info(model):
    for name, layer in model.net.input_info.items():
        log.info('\tInput blob: {}, shape: {}, precision: {}'.format(name, layer.input_data.shape, layer.precision))
    for name, layer in model.net.outputs.items():
        log.info('\tOutput blob: {}, shape: {}, precision: {}'.format(name, layer.shape, layer.precision))

def log_runtime_settings(exec_net, devices):
    if 'AUTO' not in devices:
        for device in set(parse_devices(devices)):
            try:
                nstreams = exec_net.get_config(device + '_THROUGHPUT_STREAMS')
                log.info('\tDevice: {}'.format(device))
                log.info('\t\tNumber of streams: {}'.format(nstreams))
                if device == 'CPU':
                    nthreads = exec_net.get_config('CPU_THREADS_NUM')
                    log.info('\t\tNumber of threads: {}'.format(nthreads if int(nthreads) else 'AUTO'))
            except RuntimeError:
                pass
    log.info('\tNumber of network infer requests: {}'.format(len(exec_net.requests)))
