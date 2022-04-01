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
import re
from args import DataPatternArg
from copy import deepcopy
from abc import ABC, abstractmethod

from cases import BASE, single_option_cases

THREADS_NUM = os.cpu_count()


class CorrectnessCheckerBase(ABC):
    def __init__(self, demo):
        self.filename = demo.subdirectory.replace('/', '_') + '.log'
        self.demo_name = demo.subdirectory.split('/')[0]
        self.results = {}
        self.case_index = {}

    @abstractmethod
    def __call__(self, output, test_case, device, execution_time=-1):
        print("========\nOutput: {}\n======\ntest_case: {}\n========\ndevice: {} \n==========".format(output, test_case, device))

    @abstractmethod
    def check_difference(self):
        pass

    def write_to_log(self, result, test_case, device):
        with open(self.filename, 'w') as f:
            print(self.results, file=f)


class DemoSecurityBarrierCamera(CorrectnessCheckerBase):
    def __init__(self, demo):
        super().__init__(demo)

    def __call__(self, output, test_case, device, execution_time=0):
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
        
        if execution_time < 0:
            self.case_index[device] += 1
            return

        # Parsing the raw data
        print("Demo {} results parsing....".format(self.demo_name))
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
            objid = item[2].split(':')[1]
            if objid not in self.results[device][case_index][channel][frame]:
                self.results[device][case_index][channel][frame][objid] = label_prob_pos_results
            self.results[device][case_index][channel][frame][objid] = item[3:]

        self.case_index[device] += 1
    
    def check_difference(self):
        flag = True
        devices_list = {"AUTO:GPU,CPU" : ["CPU", "GPU"],
                        "AUTO:CPU" : ["CPU"],
                        "AUTO:GPU" : ["GPU"],
                        "MULTI:GPU,CPU" : ["CPU", "GPU"]}
        err_msg = ''
        for device in devices_list:
            for target in devices_list[device]:
                if device not in self.results or target not in self.results:
                    flag = False
                    err_msg += "\tMiss the results of device {} or device {}.\n".format(device,target)
                if device in self.results and target in self.results:
                    if self.results[device] != self.results[target]:
                        flag = False
                        err_msg += "\tInconsistent results between device {} and {} \n".format(device, target)
                        # Show the detailed inconsistent results
                        for case in self.results[target]:
                            if self.results[device][case] != self.results[target][case]:
                                err_msg += ("\t\t---* Device: {} - Case: {} *----\n".format(device, case))
                                for channel in self.results[device][case]:
                                    for frame in self.results[device][case][channel]:
                                        err_msg += ("\t\t\tChannel {} - Frame {} : {}\n".format(channel, frame, self.results[device][case][channel][frame]))
                                err_msg += ('\t\t---------------------------------------------------------\n')
                                err_msg += ("\t\t---* Device: {} - Case: {} *----\n".format(target, case))
                                for channel in self.results[target][case]:
                                    for frame in self.results[target][case][channel]:
                                        err_msg += ("\t\t\tChannel {} - Frame {} : {}\n".format(channel, frame, self.results[target][case][channel][frame]))
                                err_msg += ('\t\t---------------------------------------------------------\n')
        if not flag:
            print("Correctness checking: Failure\n{}".format(err_msg))
        return flag

DEMOS = [
    deepcopy(BASE['security_barrier_camera_demo/cpp'])
    .update_option({'-r': None, '-n_iqs': '1'})
    .add_parser(DemoSecurityBarrierCamera)
]