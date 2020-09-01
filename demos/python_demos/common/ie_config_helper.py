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
    change_case = 1

    for i in range(len(device_string)):
        # For the special case of MYRIAD device names in "MYRIAD.0.0-ma0000" format
        prefix = device_string[i:i+2]
        if prefix == 'ma':
            change_case = 0
        elif prefix == 'MA' or prefix == 'Ma' or prefix == 'mA':
            change_case = -1
        elif change_case != 1 and device_string[i] == ',':
            change_case = 1

        if change_case == 1:
            formatted_string += device_string[i].upper()
        elif change_case == -1:
            formatted_string += device_string[i].lower()
        elif change_case == 0:
            formatted_string += device_string[i]

    return formatted_string


def create_config(device_string, nstreams_string, nthreads, min_latency=False):
    config = {}

    is_multi = (device_string.find('MULTI') == 0)
    devices = device_string.split(':')[-1].split(',')

    devices_nstreams = {}
    if nstreams_string:
        devices_nstreams = {device: nstreams_string for device in devices if device in device_string} \
                                    if nstreams_string.isdigit() \
                                    else dict([device.split(':') for device in nstreams_string.split(',')])

    for device in devices:
        if device == 'CPU':
            if min_latency:
                config['CPU_THROUGHPUT_STREAMS'] = '1'
                continue

            if nthreads:
                config['CPU_THREADS_NUM'] = str(nthreads)
            
            config['CPU_BIND_THREAD'] = 'NO' if is_multi and 'GPU' in devices \
                                             else 'YES'

            if 'CPU' in devices_nstreams:
                config['CPU_THROUGHPUT_STREAMS'] = devices_nstreams['CPU'] if int(devices_nstreams['CPU']) > 0 \
                                                                           else 'CPU_THROUGHPUT_AUTO'
        elif device == 'GPU':
            if min_latency:
                config['GPU_THROUGHPUT_STREAMS'] = '1'
                continue

            if 'GPU' in devices_nstreams:
                config['GPU_THROUGHPUT_STREAMS'] = devices_nstreams['GPU'] if int(devices_nstreams['GPU']) > 0 \
                                                                           else 'GPU_THROUGHPUT_AUTO'
            
            if is_multi and 'CPU' in devices:
                config['PLUGIN_THROTTLE'] = '1'
        elif 'MYRIAD' in device:
            if min_latency:
                config['MYRIAD_THROUGHPUT_STREAMS'] = '1'
                continue

            if device in devices_nstreams:
                config['MYRIAD_THROUGHPUT_STREAMS'] = devices_nstreams[device]

    return config


def create_default_config(device_string):
    config = {}

    if 'MYRIAD' in device_string:
        config['MYRIAD_THROUGHPUT_STREAMS'] = '1'

    return config
