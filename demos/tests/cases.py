# Copyright (c) 2019 Intel Corporation
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

import itertools
import sys

from args import *
from image_sequences import IMAGE_SEQUENCES

ALL_DEVICES = ['CPU', 'GPU']

TestCase = collections.namedtuple('TestCase', ['options'])

class NativeDemo:
    def __init__(self, name, test_cases):
        self._name = name

        self.test_cases = test_cases

    @property
    def full_name(self):
        return self._name

    def models_lst_path(self, source_dir):
        return source_dir / self._name / 'models.lst'

    def fixed_args(self, source_dir, build_dir):
        return [str(build_dir / self._name)]

class PythonDemo:
    def __init__(self, name, test_cases):
        self._name = name

        self.test_cases = test_cases

    @property
    def full_name(self):
        return 'py/' + self._name

    def models_lst_path(self, source_dir):
        return source_dir / 'python_demos' / self._name / 'models.lst'

    def fixed_args(self, source_dir, build_dir):
        return [sys.executable, str(source_dir / 'python_demos' / self._name / (self._name + '.py')),
            '-l', str(build_dir / 'lib/libcpu_extension.so')]

def join_cases(*args):
    options = {}
    for case in args: options.update(case.options)
    return TestCase(options=options)

def combine_cases(*args):
    return [join_cases(*combination)
        for combination in itertools.product(*[[arg] if isinstance(arg, TestCase) else arg for arg in args])]

def single_option_cases(key, *args):
    return [TestCase(options={} if arg is None else {key: arg}) for arg in args]

def device_cases(*args):
    return [TestCase(options={opt: device for opt in args}) for device in ALL_DEVICES]

NATIVE_DEMOS = [

    NativeDemo(name='smart_classroom_demo', test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            '-i': ImagePatternArg('smart-classroom-demo'),
            '-m_fd': ModelArg('face-detection-adas-0001')}),
        device_cases('-d_act', '-d_fd', '-d_lm', '-d_reid'),
        single_option_cases('-m_act', #ModelArg('person-detection-action-recognition-0005'),
                                      #ModelArg('person-detection-action-recognition-0006'),
                                      ModelArg('person-detection-raisinghand-recognition-0001')),
                                      #ModelArg('person-detection-action-recognition-teacher-0002')),
        # single_option_cases('-m_lm', ModelArg('landmarks-regression-retail-0009')),
    )),

]

PYTHON_DEMOS = [
    # TODO: 3d_segmentation_demo
    # TODO: action_recognition
    # TODO: instance_segmentation_demo
    # TODO: object_detection_demo_ssd_async
    # TODO: object_detection_demo_yolov3_async
]

DEMOS = NATIVE_DEMOS + PYTHON_DEMOS
