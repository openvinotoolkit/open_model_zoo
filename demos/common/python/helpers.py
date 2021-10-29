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

import logging as log


def resolution(value):
    try:
        result = [int(v) for v in value.split('x')]
        if len(result) != 2:
            raise RuntimeError('Сorrect format of --output_resolution parameter is "width"x"height".')
    except ValueError:
        raise RuntimeError('Сorrect format of --output_resolution parameter is "width"x"height".')
    return result

def log_layers_info(model):
    for name, metadata in model.inputs.items():
        log.info('\tInput layer: {}, shape: {}, precision: {}'.format(name, metadata.shape, metadata.precision))
    for name, metadata in model.outputs.items():
        log.info('\tOutput layer: {}, shape: {}, precision: {}'.format(name, metadata.shape, metadata.precision))

def log_runtime_settings(exec_net, devices):
    if 'AUTO' not in devices:
        for device in devices:
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

def log_latency_per_stage(*pipeline_metrics):
    stages = ('Decoding', 'Preprocessing', 'Inference', 'Postprocessing', 'Rendering')
    for stage, latency in zip(stages, pipeline_metrics):
        log.info('\t{}:\t{:.1f} ms'.format(stage, latency))
