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

from args import *
from image_sequences import IMAGE_SEQUENCES

ALL_DEVICES = ['CPU', 'GPU']

Demo = collections.namedtuple('Demo', ['name', 'test_cases'])
TestCase = collections.namedtuple('TestCase', ['options'])

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

DEMOS = [
    Demo(name='crossroad_camera_demo', test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            '-i': ImagePatternArg('person-vehicle-bike-detection-crossroad')}),
        device_cases('-d', '-d_pa', '-d_reid'),
        TestCase(options={'-m': ModelArg('person-vehicle-bike-detection-crossroad-0078')}),
        single_option_cases('-m_pa', None, ModelArg('person-attributes-recognition-crossroad-0230')),
        single_option_cases('-m_reid', None, ModelArg('person-reidentification-retail-0079')),
    )),

    Demo(name='gaze_estimation_demo', test_cases=combine_cases(
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

    Demo(name='human_pose_estimation_demo', test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            '-i': ImagePatternArg('human-pose-estimation')}),
        device_cases('-d'),
        TestCase(options={'-m': ModelArg('human-pose-estimation-0001')}),
    )),

    Demo(name='interactive_face_detection_demo', test_cases=combine_cases(
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

    # disabled because -no_show is not supported
#    Demo(name='object_detection_demo_ssd_async', test_cases=combine_cases(
#        TestCase(options={'-no_show': None}),
#        [
#            TestCase(options={
#                '-m': ModelArg('face-detection-adas-0001'),
#                '-i': ImagePatternArg('face-detection-adas-0001'),
#            }),
#            TestCase(options={
#                '-m': ModelArg('person-detection-retail-0002'),
#                '-i': ImagePatternArg('person-detection-retail-0002'),
#            }),
#            TestCase(options={
#                '-m': ModelArg('person-detection-retail-0013'),
#                '-i': ImagePatternArg('person-detection-retail-0013'),
#            }),
#        ],
#    )),

    # TODO: object_detection_demo_yolov3_async

    Demo('pedestrian_tracker_demo', test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            '-i': ImageDirectoryArg('person-detection-retail')}),
        device_cases('-d_det', '-d_reid'),
        TestCase(options={
            '-m_det': ModelArg('person-detection-retail-0013'),
            '-m_reid': ModelArg('person-reidentification-retail-0031'),
        }),
    )),

    Demo(name='security_barrier_camera_demo', test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            '-i': ImageDirectoryArg('vehicle-license-plate-detection-barrier')}),
        device_cases('-d', '-d_lpr', '-d_va'),
        TestCase(options={'-m': ModelArg('vehicle-license-plate-detection-barrier-0106')}),
        single_option_cases('-m_lpr', None, ModelArg('license-plate-recognition-barrier-0001')),
        single_option_cases('-m_va', None, ModelArg('vehicle-attributes-recognition-barrier-0039')),
    )),

    Demo(name='segmentation_demo', test_cases=combine_cases(
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

    # TODO: smart_classroom_demo

    Demo(name='super_resolution_demo', test_cases=combine_cases(
        TestCase(options={'-i': ImageDirectoryArg('single-image-super-resolution')}),
        device_cases('-d'),
        TestCase(options={
            '-m': ModelArg('single-image-super-resolution-1033'),
        }),
    )),

    Demo(name='text_detection_demo', test_cases=combine_cases(
        TestCase(options={'-no_show': None, '-dt': 'video',
            '-i': ImagePatternArg('text-detection')}),
        device_cases('-d_td', '-d_tr'),
        single_option_cases('-m_td', ModelArg('text-detection-0003'), ModelArg('text-detection-0004')),
        single_option_cases('-m_tr', None, ModelArg('text-recognition-0012')),
    )),
]
