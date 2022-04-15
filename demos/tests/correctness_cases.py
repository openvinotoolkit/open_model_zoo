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
        # Expected ROI format: lable, prob, x, y, w, h,...., Vehicle attribute, License plate 
        for item in source_roi:
            try:
                source.append(float(item))
            except ValueError:
                source.append(item)

        for item in dest_roi:
            try:
                dest.append(float(item))
            except ValueError:
                dest.append(item)
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
        err_msg = ''
        multi_correctness = {'MULTI:GPU,CPU':{'CPU': True, 'GPU': True},'AUTO:GPU,CPU':{'CPU': True, 'GPU': True}}
        multi_inconsist_results = {}
        for device in devices_list:
            tmp_msg = ''
            for target in devices_list[device]:
                if device not in self.results or target not in self.results:
                        flag = False
                        err_msg += "\tMiss the results of device {} or device {}.\n".format(device, target)
                if device in self.results and target in self.results:
                    inconsist_flag = False
                    if self.results[device] != self.results[target]:
                        tmp_msg += "\tInconsistent results between device {} and {} \n".format(device, target)
                        # Show the detailed inconsistent results
                        for case in self.results[target]:
                            if self.results[device][case] != self.results[target][case]:
                                tmp_msg += ("\t\t---* Device: {} - Case: {} *----\n".format(device, case))
                                for channel in self.results[device][case]:
                                    for frame in self.results[device][case][channel]:
                                        if channel not in self.results[target][case] or (channel in self.results[target][case] and frame not in self.results[target][case][channel]):
                                            err_msg += ("\t\t\t[Not Found on {}]Channel {} - Frame {} : {}\n".format(target, channel, frame, self.results[device][case][channel][frame]))
                                tmp_msg += ('\t\t---------------------------------------------------------\n')
                                tmp_msg += ("\t\t---* Device: {} - Case: {} *----\n".format(target, case))
                                for channel in self.results[target][case]:
                                    for frame in self.results[target][case][channel]:
                                        if channel not in self.results[device][case] or (channel in self.results[device][case] and frame not in self.results[device][case][channel]):
                                            tmp_msg += ("\t\t\t[Not Found on {}]Channel {} - Frame {} : {}\n".format(device, channel, frame, self.results[target][case][channel][frame]))
                                        else:
                                            for obj in self.results[target][case][channel][frame]:
                                                if obj not in self.results[device][case][channel][frame]:
                                                    flag = False
                                                    tmp_msg += ("\t\t\t[Not Found on {}]Channel {} - Frame {} : {}\n".format(device, channel, frame, self.results[target][case][channel][frame]))
                                                elif not self.compare_roi(self.results[device][case][channel][frame][obj],self.results[target][case][channel][frame][obj]):
                                                    if device != 'MULTI:GPU,CPU' and device != 'AUTO:GPU,CPU':
                                                        flag = False
                                                    else:
                                                        multi_correctness[device][target] = False
                                                    inconsist_flag = True
                                                    tmp_msg += ("\t\t\tInconsist result:\n\t\t\t\t[{}] Channel {} - Frame {} : {}\n".format(target, channel, frame, self.results[target][case][channel][frame]))
                                                    tmp_msg += ("\t\t\t\t[{}] Channel {} - Frame {} : {}\n".format(device, channel, frame, self.results[device][case][channel][frame]))
                                tmp_msg += ('\t\t---------------------------------------------------------\n')
                            if not inconsist_flag:
                                tmp_msg = ''
                    if inconsist_flag:
                        if device == 'MULTI:GPU,CPU' or device == 'AUTO:GPU,CPU':
                            if device not in  multi_inconsist_results:
                                multi_inconsist_results[device] = ''
                            multi_inconsist_results[device] += tmp_msg
                        else:
                            err_msg += tmp_msg
                        tmp_msg = ''
                        inconsist_flag = False
        # Check correctness for MULTI device
        for device in devices_list:
            if 'MULTI:GPU,CPU' != device and 'AUTO:GPU,CPU' != device:
                continue
            if multi_correctness[device]['CPU'] == False and multi_correctness[device]['GPU'] == False:
                flag = False
        if not flag:
            multi_msg = ''
            for device in multi_inconsist_results:
                if multi_correctness[device]['CPU'] == False and multi_correctness[device]['GPU'] == False:
                    multi_msg += multi_inconsist_results[device]
                    err_msg += multi_msg
            print("Correctness checking: Failure\n{}".format(err_msg))
        return flag

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
        #                               {"object id 0":{"label:xx,prob:xx,x,y,width,hight"},
        #                               {"object id 1":{"label:xx,prob:xx,x,y,width,hight"},
        #                               .....................
        #                               {"object id n":{"label:xx,prob:xx,x,y,width,hight"}
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
        output = [i.rstrip() for i in output.split('\n') if "DEBUG" in i and "ChannelId" in i]
        for item in output:
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
            self.results[device][case_index][channel][frame][objid] = item[3:]

        self.case_index[device] += 1

DEMOS = [
    deepcopy(BASE['security_barrier_camera_demo/cpp'])
    .update_option({'-r': None,'-ni': '16', '-n_iqs': '1', '-i': '10_images.mp4'})
    .add_parser(DemoSecurityBarrierCamera)
]

