#!/usr/bin/env python3

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

from abc import ABC, abstractmethod
from unittest import case

class Demo(ABC):
    def __init__(self, name, implementation):
        self.name = name
        self.implementation = implementation
        self.subdirectory = name + '/' + implementation
        self.results = {}
    
    @abstractmethod
    def parser(self):
        pass 

    @abstractmethod
    def checker(self):
        pass

class CPPDemo(Demo):
    def __init__(self, name, implementation='cpp'):
        super().__init__(name, implementation)
        self.results = {}
        pass

    def parser(self):
        fo = None
        try:
            fo = open('/tmp/' + self.subdirectory + '/results.log', 'r')
        except IOError as err:
            print("File error: " + str(err))
        output = [i.rstrip() for i in fo.readlines()]
        device = ''
        case_index = ''
        index = 1
        while index < len(output):
            if "Device" in output[index]:
                device = output[index][output[index].find(":") + 1:]
                if device not in self.results:
                    self.results[device] = {}

            if "CaseId" in output[index]:
                case_index = output[index][output[index].find(":") + 1:]
                if case_index not in self.results[device]:
                    self.results[device][case_index] = {}
            
            if "Execution_time" in output[index]:
                execution_time = output[index].split(':')[1]
                if execution_time == '-1':
                   while index < len(output) and 'Device' not in output[index]:
                       index += 1 
                   continue
            
            # Pase the raw data
            while index < len(output) and 'ChannelId' in output[index]:
                item = output[index][output[index].find('ChannelId'):].split(',')
                # Channel ID
                frame_results = {}
                channel = item[0].split(':')
                if channel[1] not in self.results[device][case_index]:
                    self.results[device][case_index][channel[1]] = frame_results

                # Frame ID
                object_results = {}
                frame = item[1].split(':')
                if frame[1] not in self.results[device][case_index][channel[1]]:
                    self.results[device][case_index][channel[1]][frame[1]] = object_results

                # Object ID
                label_prob_pos_results = []
                objid = item[2].split(':')
                if objid[1] not in self.results[device][case_index][channel[1]][frame[1]]:
                    self.results[device][case_index][channel[1]][frame[1]][objid[1]] = label_prob_pos_results
                self.results[device][case_index][channel[1]][frame[1]][objid[1]] = item[3:]
                index += 1

            index += 1

    
    def checker(self):
        self.parser()
        for device in self.results:
            for case in self.results[device]:
                print ("---* Device: {} - Case: {} *----\n".format(device, case))
                for channel in self.results[device][case]:
                    print ("Channel: {} - :{}".format(channel, self.results[device][case][channel]))
                print ('---------------------------------------------------------')

        flag = False
        if "CPU" in self.results and "AUTO:CPU" in self.results:
            if self.results['CPU'] == self.results['AUTO:CPU']:
                flag = True
            else:
                print ("CPU vs AUTO:CPU have inconsistent results")

        if "GPU" in self.results and "AUTO:GPU" in self.results:
            if self.results['GPU'] == self.results['AUTO:GPU']:
                flag = True
            else:
                print ("GPU vs AUTO:GPU have inconsistent results")

        return flag

    

DEMOS = [
    CPPDemo(name='security_barrier_camera_demo')
]
def main():
   for demo in DEMOS:
       print ("Checking {}...".format(demo.name))
       if demo.checker():
           print("Demo: {} passed.".format(demo.name))
       else:
           print("Demo: {} failed.".format(demo.name))


if __name__ == '__main__':
    main()