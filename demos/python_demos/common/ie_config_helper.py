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

def format_device_string(device_string):
    formatted_string = ''
    preserve_case = False

    for i in range(len(device_string)):
        # These two conditions handle the special case of MYRIAD device names in "ma1234" format, where letters should
        # not be transformed to upper case.
        # if device_string[i] == 'm' and i+1 != len(device_string) and device_string[i+1] == 'a':
        #     preserve_case = True
        if preserve_case and device_string[i] == ',':
            preserve_case = False

        formatted_string += device_string[i] if preserve_case \
                                             else device_string[i].upper()
    
    return formatted_string


def create_config(device_string, nstreams_string, nthreads, min_latency=False):
    config = {}

    devices_nstreams = {}
    if nstreams_string:
        devices_nstreams = {device: nstreams_string for device in ['CPU', 'GPU', 'MYRIAD'] if device in device_string} \
                                    if nstreams_string.isdigit() \
                                    else dict([device.split(':') for device in nstreams_string.split(',')])

    if 'CPU' in device_string:
        if min_latency:
            config['CPU_THROUGHPUT_STREAMS'] = '1'
        else:
            if nthreads:
                config['CPU_THREADS_NUM'] = str(nthreads)

            if 'CPU' in devices_nstreams:
                config['CPU_THROUGHPUT_STREAMS'] = devices_nstreams['CPU'] if int(devices_nstreams['CPU']) > 0 \
                                                                           else 'CPU_THROUGHPUT_AUTO'
    
    if 'GPU' in device_string:
        if min_latency:
            config['GPU_THROUGHPUT_STREAMS'] = '1'
        elif 'GPU' in devices_nstreams:
            config['GPU_THROUGHPUT_STREAMS'] = devices_nstreams['GPU'] if int(devices_nstreams['GPU']) > 0 \
                                                                       else 'GPU_THROUGHPUT_AUTO'
    
    if 'MYRIAD' in device_string: #or 'ma' in device_string:
        if min_latency:
            config['MYRIAD_THROUGHPUT_STREAMS'] = '1'
        # elif devices_nstreams.count('MYRIAD') > 0 or :
        #         config.insert({ InferenceEngine::MYRIAD_THROUGHPUT_STREAMS,
        #                         std::to_string(deviceNstreams.at(device)) });

    return config


def create_default_config(device_string):
    config = {}

    if 'MYRIAD' in device_string:
        config['MYRIAD_THROUGHPUT_STREAMS'] = '1'

    return config
