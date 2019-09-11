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
    NativeDemo(name='crossroad_camera_demo', test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            '-i': ImagePatternArg('person-vehicle-bike-detection-crossroad')}),
        device_cases('-d', '-d_pa', '-d_reid'),
        TestCase(options={'-m': ModelArg('person-vehicle-bike-detection-crossroad-0078')}),
        single_option_cases('-m_pa', None, ModelArg('person-attributes-recognition-crossroad-0230')),
        single_option_cases('-m_reid', None, ModelArg('person-reidentification-retail-0079')),
    )),

    NativeDemo(name='gaze_estimation_demo', test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            '-i': ImagePatternArg('gaze-estimation-adas')}),
        device_cases('-d', '-d_fd', '-d_hp', '-d_lm'),
        TestCase(options={
            '-m': ModelArg('gaze-estimation-adas-0002'),
            '-m_fd': ModelArg('face-detection-adas-0001'),
            '-m_hp': ModelArg('head-pose-estimation-adas-0001'),
            '-m_lm': ModelArg('facial-landmarks-35-adas-0002'),
        }),
    )),

    NativeDemo(name='human_pose_estimation_demo', test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            '-i': ImagePatternArg('human-pose-estimation')}),
        device_cases('-d'),
        TestCase(options={'-m': ModelArg('human-pose-estimation-0001')}),
    )),

    NativeDemo(name='interactive_face_detection_demo', test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            '-i': ImagePatternArg('face-detection-adas')}),
        device_cases('-d', '-d_ag', '-d_em', '-d_lm', '-d_hp'),
        TestCase(options={'-m': ModelArg('face-detection-adas-0001')}),
        [
            TestCase(options={}),
            TestCase(options={'-m_ag': ModelArg('age-gender-recognition-retail-0013')}),
            TestCase(options={'-m_em': ModelArg('emotions-recognition-retail-0003')}),
            TestCase(options={'-m_lm': ModelArg('facial-landmarks-35-adas-0002')}),
            TestCase(options={'-m_hp': ModelArg('head-pose-estimation-adas-0001')}),
            TestCase(options={
                '-m_ag': ModelArg('age-gender-recognition-retail-0013'),
                '-m_em': ModelArg('emotions-recognition-retail-0003'),
                '-m_hp': ModelArg('head-pose-estimation-adas-0001'),
                '-m_lm': ModelArg('facial-landmarks-35-adas-0002'),
            })
        ],
    )),

    # TODO: mask_rcnn_demo

    # TODO: multichannel demos

    # TODO: object_detection_demo_faster_rcnn

    NativeDemo(name='object_detection_demo_ssd_async', test_cases=combine_cases(
        TestCase(options={'-no_show': None}),
        [
            TestCase(options={
                '-m': ModelArg('face-detection-adas-0001'),
                '-i': ImagePatternArg('face-detection-adas'),
            }),
            TestCase(options={
                '-m': ModelArg('person-detection-retail-0002'),
                '-i': ImagePatternArg('person-detection-retail'),
            }),
            TestCase(options={
                '-m': ModelArg('person-detection-retail-0013'),
                '-i': ImagePatternArg('person-detection-retail'),
            }),
        ],
    )),

    # TODO: object_detection_demo_yolov3_async

    NativeDemo('pedestrian_tracker_demo', test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            '-i': ImageDirectoryArg('person-detection-retail')}),
        device_cases('-d_det', '-d_reid'),
        [
            TestCase(options={'-m_det': ModelArg('person-detection-retail-0002')}),
            TestCase(options={'-m_det': ModelArg('person-detection-retail-0013')}),
        ],
        single_option_cases('-m_reid',
            ModelArg('person-reidentification-retail-0031'),
            ModelArg('person-reidentification-retail-0076'),
            ModelArg('person-reidentification-retail-0079')),
    )),

    NativeDemo(name='security_barrier_camera_demo', test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            '-i': ImageDirectoryArg('vehicle-license-plate-detection-barrier')}),
        device_cases('-d', '-d_lpr', '-d_va'),
        TestCase(options={'-m': ModelArg('vehicle-license-plate-detection-barrier-0106')}),
        single_option_cases('-m_lpr', None, ModelArg('license-plate-recognition-barrier-0001')),
        single_option_cases('-m_va', None, ModelArg('vehicle-attributes-recognition-barrier-0039')),
    )),

    NativeDemo(name='segmentation_demo', test_cases=combine_cases(
        device_cases('-d'),
        [
            TestCase(options={
                '-m': ModelArg('road-segmentation-adas-0001'),
                '-i': ImageDirectoryArg('road-segmentation-adas'),
            }),
            TestCase(options={
                '-m': ModelArg('semantic-segmentation-adas-0001'),
                '-i': ImageDirectoryArg('semantic-segmentation-adas'),
            }),
        ],
    )),

    NativeDemo(name='smart_classroom_demo', test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            '-i': ImagePatternArg('smart-classroom-demo'),
            '-m_fd': ModelArg('face-detection-adas-0001')}),
        device_cases('-d_act', '-d_fd', '-d_lm', '-d_reid'),
        single_option_cases('-m_act', ModelArg('person-detection-action-recognition-0005'),
                                      ModelArg('person-detection-action-recognition-0006'),
                                      ModelArg('person-detection-raisinghand-recognition-0001'),
                                      ModelArg('person-detection-action-recognition-teacher-0002')),
        single_option_cases('-m_lm', None, ModelArg('landmarks-regression-retail-0009')),
        single_option_cases('-m_reid', None, ModelArg('face-reidentification-retail-0095')),
    )),

    NativeDemo(name='super_resolution_demo', test_cases=combine_cases(
        TestCase(options={'-i': ImageDirectoryArg('single-image-super-resolution')}),
        device_cases('-d'),
        TestCase(options={
            '-m': ModelArg('single-image-super-resolution-1033'),
        }),
    )),

    NativeDemo(name='text_detection_demo', test_cases=combine_cases(
        TestCase(options={'-no_show': None, '-dt': 'video',
            '-i': ImagePatternArg('text-detection')}),
        device_cases('-d_td', '-d_tr'),
        single_option_cases('-m_td', ModelArg('text-detection-0003'), ModelArg('text-detection-0004')),
        single_option_cases('-m_tr', None, ModelArg('text-recognition-0012')),
    )),
]

PYTHON_DEMOS = [
    # TODO: 3d_segmentation_demo
    # TODO: action_recognition
    # TODO: instance_segmentation_demo
    # TODO: object_detection_demo_ssd_async
    # TODO: object_detection_demo_yolov3_async

    PythonDemo(name='segmentation_demo', test_cases=combine_cases(
        device_cases('-d'),
        [
            TestCase(options={
                '-m': ModelArg('road-segmentation-adas-0001'),
                '-i': IMAGE_SEQUENCES['road-segmentation-adas'],
            }),
            TestCase(options={
                '-m': ModelArg('semantic-segmentation-adas-0001'),
                '-i': IMAGE_SEQUENCES['semantic-segmentation-adas'],
            }),
        ],
    )),
]

DEMOS = NATIVE_DEMOS + PYTHON_DEMOS
