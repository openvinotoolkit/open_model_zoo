# Copyright (c) 2021 Intel Corporation
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

import os
from copy import deepcopy
from abc import ABC, abstractmethod

from cases import BASE

class CorrectnessCheckerBase(ABC):
    def __init__(self, demo):
        self.filename = demo.subdirectory.replace('/', '_') + '.log'
        self.demo_name = demo.subdirectory.split('/')[0]
        self.results = {}
        self.case_index = {}

    @abstractmethod
    def __call__(self, output, test_case, device, execution_time=-1):
        pass

    def compare_roi(self, source_roi, dest_roi):
        source = []
        dest = []
        if len(source_roi) != len(dest_roi):
            return False
        # Expected ROI format: able, prob, x, y, w, h,....
        for item in source_roi:
            source.append(float(item))

        for item in dest_roi:
            dest.append(float(item))

        flag = True
        prob_gap = 0.1
        pos_gap = 20
        for index in range(len(source)):
            if type(source[index]) == float:
                if source[index] > 0 and source[index] < 1:
                    if abs(source[index] - dest[index]) > prob_gap:
                        flag = False
                        break
                else:
                    if abs(source[index] - dest[index]) > pos_gap:
                        flag = False
                        break
            else:
                if source[index] != dest[index]:
                    flag = False
                    break

        return flag

    def check_difference(self):
        flag = True
        devices_list = {
                        "AUTO:GPU,CPU" : ["CPU", "GPU"],
                        "MULTI:GPU,CPU" : ["CPU", "GPU"],
                        "AUTO:CPU" : ["CPU"],
                        "AUTO:GPU" : ["GPU"],
                        }
        # Record the results for the single device inputting like AUTO:CPU, AUTO:GPU
        # {
        #   device 0: { target 0: {case 0: True/False, case 1: True/False, ...},...,target n:{} },
        #   device 1: { target 1: {case 0: True/False, case 1: True/False, ...},...,target n:{} },
        # }
        multi_correctness = {}

        # Record the results for the multi devices inputting like AUTO:GPU,CPU, MULTI:GPU,CPU
        #{
        #   device 0:{
        #          case 0: {
        #                    channel 0: {
        #                                   frame 0: [roi_result, attribute_result, license_plate_results],
        #                                   frame 1: [roi_result, attribute_result, license_plate_results],
        #                                   ...
        #                               },
        #                    channel 1: {},
        #                    .........
        #                   },
        #           case 1:{},
        #           .......
        #           }
        #   device 1:{},
        #}
        multi_correctness_devices = {}

        # Record the detailed msg for the inconsistent results
        # {
        #   device 0: {case 0: "error msg", case 1: "error msg", ...}, 
        #   device 1: {case 0: "error msg", case 1: "error msg", ...}, 
        #   ....
        # }
        multi_correctness_errmsg = {}
        for device in devices_list:
            multi_correctness[device] = {}
            multi_correctness_errmsg[device] = {}
            if 'GPU' in device and 'CPU' in device:
                multi_correctness_devices[device] = {}
            for target in devices_list[device]:
                multi_correctness[device][target] = {}
                if 'GPU' in device and 'CPU' in device:
                    multi_correctness_devices[device][target] = {}
        for device in devices_list:
            for target in devices_list[device]:
                if device not in self.results or target not in self.results:
                    multi_correctness[device][target] = {}
                    multi_correctness_errmsg[device]['-1'] = "\tMiss the results from device {} or from device {}.\n".format(device, target)
                    continue
                #if self.results[device] != self.results[target]:
                #    # Show the detailed inconsistent results
                for case in self.results[target]:
                    if case not in multi_correctness[device][target]:
                        multi_correctness[device][target][case] = True 
                    if case not in multi_correctness_errmsg[device]:
                        multi_correctness_errmsg[device][case] = ''

                    #if self.results[device][case] != self.results[target][case]:
                    for channel in self.results[device][case]:
                        for frame in self.results[device][case][channel]:
                            if channel not in self.results[target][case] or (channel in self.results[target][case] and frame not in self.results[target][case][channel]):
                                multi_correctness[device][target][case] = False
                                multi_correctness_errmsg[device][case] += "[Device: {}- Case: {}][Not Found on {}]Channel {} - Frame {} : {}\n".format(device, case, target, channel, frame, self.results[device][case][channel][frame])
                            else:
                                for obj in self.results[target][case][channel][frame]:
                                    if obj not in self.results[device][case][channel][frame]:
                                        multi_correctness[device][target][case] = False
                                        multi_correctness_errmsg[device][case] += "[Device: {}- Case: {}][Not Found on {}]Channel {} - Frame {} : {}\n".format(device, case, device, channel, frame, self.results[target][case][channel][frame])
                                    else:
                                        try:
                                            if 'CPU' in device and 'GPU' in device:
                                                if case not in multi_correctness_devices[device][target]:
                                                    multi_correctness_devices[device][target][case] = {}
                                                if channel not in multi_correctness_devices[device][target][case]:
                                                    multi_correctness_devices[device][target][case][channel] = {}
                                                if frame not in multi_correctness_devices[device][target][case][channel]:
                                                    multi_correctness_devices[device][target][case][channel][frame] = [] 

                                            for i in range(len(self.results[device][case][channel][frame][obj])):
                                                if i == 0:
                                                    # Compared ROI
                                                    device_vehicle_roi = self.results[device][case][channel][frame][obj][i] 
                                                    target_vehicle_roi = self.results[target][case][channel][frame][obj][i] 
                                                    flag_roi = self.compare_roi(device_vehicle_roi, target_vehicle_roi)
                                                    if 'CPU' in device and 'GPU' in device:
                                                        multi_correctness_devices[device][target][case][channel][frame].append(flag_roi)
                                                    else:
                                                        if not flag_roi: 
                                                            multi_correctness[device][target][case] = False 
                                                            tmp_msg = ("[Device: {}- Case: {} on {}] Channel {} - Frame {} : {}\n".format(device, case, device, channel, frame, self.results[device][case][channel][frame]))
                                                            tmp_msg += ("[Device: {}- Case: {} on {}] Channel {} - Frame {} : {}\n".format(device, case, target, channel, frame, self.results[target][case][channel][frame]))
                                                            multi_correctness_errmsg[device][case] += tmp_msg
                                                else:
                                                    # Compare attribute/license plate
                                                    device_vehicle_attr = self.results[device][case][channel][frame][obj][i]
                                                    target_vehicle_attr = []
                                                    if i < len(self.results[target][case][channel][frame][obj]):
                                                        target_vehicle_attr = self.results[target][case][channel][frame][obj][i]
                                                    if 'CPU' in device and 'GPU' in device:
                                                        multi_correctness_devices[device][target][case][channel][frame].append(device_vehicle_attr == target_vehicle_attr)
                                                    else:
                                                        if device_vehicle_attr != target_vehicle_attr: 
                                                            multi_correctness[device][target][case] = False 
                                                            tmp_msg = ("[Device: {}- Case: {} on {}] Channel {} - Frame {} : {}\n".format(device, case, device, channel, frame, self.results[device][case][channel][frame]))
                                                            tmp_msg += ("[Device: {}- Case: {} on {}] Channel {} - Frame {} : {}\n".format(device, case, target, channel, frame, self.results[target][case][channel][frame]))
                                                            multi_correctness_errmsg[device][case] += tmp_msg
                                            #if 'CPU' in device and 'GPU' in device:
                                            #    print(multi_correctness_devices[device][target][case][channel][frame])
                                            #else:
                                            #    print(multi_correctness[device][target][case])
                                            #print("======================================")
                                            if 'CPU' in device and 'GPU' in device:
                                                # Check if correctness result between device and target for multi devices
                                                consistent_flag = False 
                                                for flag in multi_correctness_devices[device][target][case][channel][frame]:
                                                    if flag == True:
                                                        consistent_flag = True
                                                        break
                                                if not consistent_flag:
                                                        tmp_msg = ("[Device: {}- Case: {} on {}] Channel {} - Frame {} : {}\n".format(device, case, target, channel, frame, self.results[target][case][channel][frame]))
                                                        if 'CPU' in device and 'GPU' in device and target == 'CPU':
                                                            tmp_msg += ("[Device: {}- Case: {} on {}] Channel {} - Frame {} : {}\n".format(device, case, 'GPU', channel, frame, self.results['GPU'][case][channel][frame]))
                                                        elif 'CPU' in device and 'GPU' in device and target == 'GPU':
                                                            tmp_msg += ("[Device: {}- Case: {} on {}] Channel {} - Frame {} : {}\n".format(device, case, 'CPU', channel, frame, self.results['CPU'][case][channel][frame]))
                                                        tmp_msg += ("[Device: {}- Case: {} on {}] Channel {} - Frame {} : {}\n".format(device, case, device, channel, frame, self.results[device][case][channel][frame]))
                                                        multi_correctness_errmsg[device][case] += tmp_msg
                                        except:
                                            print("======Checking exception on Case:{} - Device: {} - Target: {} - channel: {} - frame: {}=======".format(case, device, target, channel, frame))
                                            print("Device {}: {}".format(device, self.results[device][case][channel][frame][obj]))
                                            print("{}: {}".format(target, self.results[target][case][channel][frame][obj]))
                                            if 'CPU' in device and 'GPU' in device:
                                                multi_correctness_devices[device][target][case][channel][frame].append(False)
                                                if target == 'CPU':
                                                    print("GPU: {}".format(self.results['GPU'][case][channel][frame][obj]))
                                                elif target == 'GPU':
                                                    print("CPU: {}".format(self.results['CPU'][case][channel][frame][obj]))
                                            else:
                                                multi_correctness[device][target][case] = False
                                                multi_correctness_errmsg[device][case] += "[Device: {}- Case: {}][Exception on {}]Channel {} - Frame {} : {}\n".format(device, case, device, channel, frame, self.results[target][case][channel][frame])



        #print("=====================================")
        #print(multi_correctness)
        #print("=====================================")
        #print(multi_correctness_errmsg)
        #print("=====================================")
        #for device in multi_correctness_devices: 
        #    for target in multi_correctness_devices[device]:
        #        for case in multi_correctness_devices[device][target]:
        #            for channel in multi_correctness_devices[device][target][case]: 
        #                print("===== Device:{} - target: {} - Case: {} - channel: {}=====".format(device, target, case, channel))
        #                print(multi_correctness_devices[device][target][case][channel])
        #print("=====================================")
        final_correctness_flag = True 
        for device in devices_list:
            if 'MULTI:GPU,CPU' != device and 'AUTO:GPU,CPU' != device:
                for target in multi_correctness[device]:
                    consistent_flag = True
                    err_msg = 'Inconsistent result:\n'
                    if len(multi_correctness[device][target]) == 0:
                        final_correctness_flag = False 
                        consistent_flag = False
                        # Miss results for device
                        err_msg += '\t'
                        err_msg += multi_correctness_errmsg[device]['-1'] 
                    else:
                        for case in multi_correctness[device][target]:
                            flag = multi_correctness[device][target][case]
                            if flag == False:
                                final_correctness_flag = False
                                consistent_flag = False
                                err_msg += '\t'
                                err_msg += multi_correctness_errmsg[device][case] 
                    if not consistent_flag:
                        print("Checking device: {} - target : {} - : Fail.\n{}".format(device, devices_list[device], err_msg))
                    else:
                        print("Checking device: {} - target : {} - : PASS.\n".format(device, devices_list[device]))
            else:
                consistent_flag = True
                err_msg = ''
                if len(multi_correctness[device][target]) == 0:
                    final_correctness_flag = False 
                    consistent_flag = False
                    # Miss results for device
                    err_msg += multi_correctness_errmsg[device]['-1'] 
                else:
                    for case in multi_correctness[device]['CPU']:
                        if multi_correctness[device]['CPU'][case] == False:
                            final_correctness_flag = False
                            consistent_flag = False 
                            err_msg += multi_correctness_errmsg[device][case] 
                            #print(multi_correctness_errmsg[device][target][case])
                        else:
                            for channel in multi_correctness_devices[device]['GPU'][case]:
                                for frame in multi_correctness_devices[device]['GPU'][case][channel]:
                                    frame_flag = True
                                    for i in range(len(multi_correctness_devices[device]['GPU'][case][channel][frame])):
                                        gpu_frame_consistent_flag = multi_correctness_devices[device]['GPU'][case][channel][frame][i]
                                        cpu_frame_consistent_flag = multi_correctness_devices[device]['CPU'][case][channel][frame][i]
                                        if gpu_frame_consistent_flag == False and cpu_frame_consistent_flag == False:
                                            frame_flag = False 
                                    #print("checking result: {}\nDevice : GPU - case: {} - channel: {} - frame: {}: {}".format(frame_flag, case, channel, frame, multi_correctness_devices[device]['GPU'][case][channel][frame]))
                                    #print("Device : CPU - case: {} - channel: {} - frame: {}: {}".format(case, channel, frame, multi_correctness_devices[device]['CPU'][case][channel][frame]))
                                    if not frame_flag:
                                        final_correctness_flag = False
                                        consistent_flag = False 
                                        err_msg += "Inconsistent result:\n\tDevice: {} - case: {} - channel: {} - frame: {}: {}\n".format(device, case, channel, frame, self.results[device][case][channel][frame])
                                        for target in devices_list[device]:
                                            err_msg += "\tDevice: {} - case: {} - channel: {} - frame: {}: {}\n".format(target, case, channel, frame, self.results[target][case][channel][frame])
                                        for target in devices_list[device]:
                                            err_msg += "\tDevice: {} - case: {} - channel: {} - frame: {}: {}\n".format(target, case, channel, frame, multi_correctness_devices[device][target][case][channel][frame]) 

                if not consistent_flag:
                    print("Checking device: {} - target : {} - : Fail.\n{}".format(device, devices_list[device], err_msg))
                else:
                    print("Checking device: {} - target : {} - : PASS.\n".format(device, devices_list[device]))
        return final_correctness_flag 

    def write_to_log(self, result, test_case, device):
        with open(self.filename, 'w') as f:
            print(self.results, file=f)


class DemoSecurityBarrierCamera(CorrectnessCheckerBase):
    def __call__(self, output, test_case, device, execution_time=0):
        # Parsing results from raw data
        # Results format
        #               {"device name":
        #                   {"case index 0":
        #                       {"channel id 0":
        #                           {"frame id 0":
        #                               {"object id 0":[["label", "prob", "x", "y", "width", "hight"],"Vehicle attribute", "License plate"],
        #                               {"object id 1":["label", "prob", "x", "y", "width", "hight","Vehicle attribute", "License plate"],
        #                               .....................
        #                               {"object id n":["label", "prob", "x", "y", "width", "hight","Vehicle attribute", "License plate"],
        #                           },
        #                           .....
        #                           {"frame id n":
        #                               .....
        #                           }
        #                       },
        #                       .....
        #                       {"channel id n":
        #                       .....
        #                       }
        #                   {"case index n":
        #                       .....
        #                   }
        #               }
        # Generate case id for each device
        if device not in self.case_index:
            self.case_index[device] = 0

        if device not in self.results:
            self.results[device] = {}

        case_index = self.case_index[device]
        if case_index not in self.results[device]:
            self.results[device][case_index] = {}

        # Parsing the raw data
        try:
            output = [i.rstrip() for i in output.split('\n') if "DEBUG" in i and "ChannelId" in i]
            for item in output:
                line = item
                item = item[item.find('ChannelId'):].split(',')
                # Channel ID
                frame_results = {}
                channel = item[0].split(':')[1]
                if channel not in self.results[device][case_index]:
                    self.results[device][case_index][channel] = frame_results

                # Frame ID
                object_results = {}
                frame = item[1].split(':')[1]
                if frame not in self.results[device][case_index][channel]:
                    self.results[device][case_index][channel][frame] = object_results

                # Object ID
                label_prob_pos_results = []
                objid = item[2]
                if objid not in self.results[device][case_index][channel][frame]:
                    self.results[device][case_index][channel][frame][objid] = label_prob_pos_results
                #self.results[device][case_index][channel][frame][objid] = item[3:]
                roi_attr_license = ",".join(item[3:]).split('\t')
                for index in range(len(roi_attr_license)):
                    if index == 0:
                        item = roi_attr_license[index].split(',')
                        self.results[device][case_index][channel][frame][objid].append(item)
                    else:
                        self.results[device][case_index][channel][frame][objid].append(roi_attr_license[index])
                #print("Device: {}- case: {} - channel: {} - frame: {}: {}".format(device, case_index, channel, frame, self.results[device][case_index][channel][frame]))
        except IndexError:
            raise IndexError ("Rawdata format is invalid:\n\t{}".format(line))
            

        self.case_index[device] += 1

DEMOS = [
    deepcopy(BASE['security_barrier_camera_demo/cpp'])
    .update_option({'-r': None,'-ni': '16', '-n_iqs': '1', '-i': 'multi_images.mp4'})
    .add_parser(DemoSecurityBarrierCamera)
]

