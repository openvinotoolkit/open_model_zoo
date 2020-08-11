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
from data_sequences import DATA_SEQUENCES

MONITORS = {'-u': 'cdm'}
TestCase = collections.namedtuple('TestCase', ['options'])

class Demo:

    def device_args(self, device_list):
        if len(self.device_keys) == 0:
            return {'CPU': []}
        return {device: [arg for key in self.device_keys for arg in [key, device]] for device in device_list}

class NativeDemo(Demo):
    def __init__(self, subdirectory, device_keys, test_cases):
        self.subdirectory = subdirectory

        self.device_keys = device_keys

        self.test_cases = test_cases

        self._name = subdirectory.replace('/', '_')

    @property
    def full_name(self):
        return self._name

    def models_lst_path(self, source_dir):
        return source_dir / self.subdirectory / 'models.lst'

    def fixed_args(self, source_dir, build_dir):
        return [str(build_dir / self._name)]

class PythonDemo(Demo):
    def __init__(self, subdirectory, device_keys, test_cases):
        self.subdirectory = 'python_demos/' + subdirectory

        self.device_keys = device_keys

        self.test_cases = test_cases

        self._name = subdirectory.replace('/', '_')

    @property
    def full_name(self):
        return 'py/' + self._name

    def models_lst_path(self, source_dir):
        return source_dir / self.subdirectory / 'models.lst'

    def fixed_args(self, source_dir, build_dir):
        cpu_extension_path = build_dir / 'lib/libcpu_extension.so'

        return [sys.executable, str(source_dir / 'python_demos' / self._name / (self._name + '.py')),
            *(['-l', str(cpu_extension_path)] if cpu_extension_path.exists() else [])]

def join_cases(*args):
    options = {}
    for case in args: options.update(case.options)
    return TestCase(options=options)

def combine_cases(*args):
    return [join_cases(*combination)
        for combination in itertools.product(*[[arg] if isinstance(arg, TestCase) else arg for arg in args])]

def single_option_cases(key, *args):
    return [TestCase(options={} if arg is None else {key: arg}) for arg in args]


NATIVE_DEMOS = [
    NativeDemo(subdirectory='crossroad_camera_demo',
            device_keys=['-d', '-d_pa', '-d_reid'],
            test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            **MONITORS,
            '-i': DataPatternArg('person-vehicle-bike-detection-crossroad')}),
        TestCase(options={'-m': ModelArg('person-vehicle-bike-detection-crossroad-0078')}),
        single_option_cases('-m_pa', None, ModelArg('person-attributes-recognition-crossroad-0230')),
        single_option_cases('-m_reid',
            None,
            ModelArg('person-reidentification-retail-0248'),
            ModelArg('person-reidentification-retail-0265'),
            ModelArg('person-reidentification-retail-0267'),
            ModelArg('person-reidentification-retail-0270')),
    )),

    NativeDemo(subdirectory='gaze_estimation_demo',
            device_keys=['-d', '-d_fd', '-d_hp', '-d_lm'],
            test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            **MONITORS,
            '-i': DataPatternArg('gaze-estimation-adas')}),
        TestCase(options={
            '-m': ModelArg('gaze-estimation-adas-0002'),
            '-m_fd': ModelArg('face-detection-adas-0001'),
            '-m_hp': ModelArg('head-pose-estimation-adas-0001'),
            '-m_lm': ModelArg('facial-landmarks-35-adas-0002'),
            '-m_es': ModelArg('open-closed-eye-0001'),
        }),
    )),

    NativeDemo(subdirectory='human_pose_estimation_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            **MONITORS,
            '-i': DataPatternArg('human-pose-estimation')}),
        TestCase(options={'-m': ModelArg('human-pose-estimation-0001')}),
    )),

    NativeDemo(subdirectory='classification_demo',
            device_keys=['-d'],
            test_cases=combine_cases(
        TestCase(options={
            '-no_show': None,
            '-time': '5',
            '-i': DataDirectoryOrigFileNamesArg('classification'),
            '-labels': DemoFileArg('imagenet_2012_classes.txt'),
            '-gt': TestDataArg("ILSVRC2012_img_val/ILSVRC2012_val.txt"),
            '-b': '8'}),
        single_option_cases('-m',
            ModelArg('alexnet'),
            ModelArg('densenet-121-tf'),
            ModelArg('densenet-169'),
            ModelArg('mobilenet-v2-pytorch'),
            ModelArg('resnet-50')),
    )),

    NativeDemo(subdirectory='interactive_face_detection_demo',
            device_keys=['-d', '-d_ag', '-d_em', '-d_lm', '-d_hp'],
            test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            **MONITORS,
            '-i': DataPatternArg('face-detection-adas')}),
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

    NativeDemo(subdirectory='mask_rcnn_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'-i': DataDirectoryArg('semantic-segmentation-adas')}),
        single_option_cases('-m',
            ModelArg('mask_rcnn_inception_resnet_v2_atrous_coco'),
            ModelArg('mask_rcnn_inception_v2_coco'),
            ModelArg('mask_rcnn_resnet101_atrous_coco'),
            ModelArg('mask_rcnn_resnet50_atrous_coco'))
    )),

    NativeDemo(subdirectory='multi_channel/face_detection_demo',
            device_keys=['-d'],
            test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            **MONITORS,
            '-i': DATA_SEQUENCES['face-detection-adas']}),
        single_option_cases('-m',
            ModelArg('face-detection-adas-0001'),
            ModelArg('face-detection-adas-binary-0001', "FP32-INT1"),
            ModelArg('face-detection-retail-0004'),
            ModelArg('face-detection-retail-0005'),
            ModelArg('face-detection-retail-0044')),
    )),

    NativeDemo(subdirectory='multi_channel/human_pose_estimation_demo', device_keys=['-d'],
            test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            **MONITORS,
            '-i': DATA_SEQUENCES['human-pose-estimation'],
            '-m': ModelArg('human-pose-estimation-0001')}),
    )),

    NativeDemo(subdirectory='object_detection_demo_ssd_async', device_keys=[], test_cases=combine_cases(
        TestCase(options={'-no_show': None, **MONITORS}),
        [
            TestCase(options={
                '-m': ModelArg('face-detection-adas-0001'),
                '-i': DataPatternArg('face-detection-adas'),
            }),
            TestCase(options={
                '-m': ModelArg('person-detection-retail-0002'),
                '-i': DataPatternArg('person-detection-retail'),
            }),
            TestCase(options={
                '-m': ModelArg('person-detection-retail-0013'),
                '-i': DataPatternArg('person-detection-retail'),
            }),
        ],
    )),

    NativeDemo(subdirectory='object_detection_demo_yolov3_async', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'--no_show': None,
            **MONITORS,
            '-i': DataPatternArg('object-detection-demo-ssd-async')}),
        TestCase(options={'-m': ModelArg('yolo-v3-tf')})
    )),

    NativeDemo('pedestrian_tracker_demo', device_keys=['-d_det', '-d_reid'], test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            **MONITORS,
            '-i': DataPatternArg('person-detection-retail')}),
        [
            TestCase(options={'-m_det': ModelArg('person-detection-retail-0002')}),
            TestCase(options={'-m_det': ModelArg('person-detection-retail-0013')}),
        ],
        single_option_cases('-m_reid',
            ModelArg('person-reidentification-retail-0248'),
            ModelArg('person-reidentification-retail-0265'),
            ModelArg('person-reidentification-retail-0267'),
            ModelArg('person-reidentification-retail-0270')),
    )),

    NativeDemo(subdirectory='security_barrier_camera_demo',
            device_keys=['-d', '-d_lpr', '-d_va'],
            test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            **MONITORS,
            '-i': DataDirectoryArg('vehicle-license-plate-detection-barrier')}),
        TestCase(options={'-m': ModelArg('vehicle-license-plate-detection-barrier-0106')}),
        single_option_cases('-m_lpr',
            None,
            ModelArg('license-plate-recognition-barrier-0001'),
            ModelArg('license-plate-recognition-barrier-0007')),
        single_option_cases('-m_va', None, ModelArg('vehicle-attributes-recognition-barrier-0039')),
    )),

    NativeDemo(subdirectory='segmentation_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'-no_show': None, **MONITORS}),
        [
            TestCase(options={
                '-m': ModelArg('road-segmentation-adas-0001'),
                '-i': DataPatternArg('road-segmentation-adas'),
            }),
            *combine_cases(
                TestCase(options={'-i': DataPatternArg('semantic-segmentation-adas')}),
                single_option_cases('-m',
                    ModelArg('semantic-segmentation-adas-0001'),
                    ModelArg('deeplabv3'))),
        ],
    )),

    NativeDemo(subdirectory='smart_classroom_demo',
            device_keys=['-d_act', '-d_fd', '-d_lm', '-d_reid'],
            test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            **MONITORS,
            '-i': DataPatternArg('smart-classroom-demo'),
            '-m_fd': ModelArg('face-detection-adas-0001')}),
        [
            *combine_cases(
                [
                    TestCase(options={'-m_act': ModelArg('person-detection-action-recognition-0005')}),
                    TestCase(options={'-m_act': ModelArg('person-detection-action-recognition-0006'),
                        '-student_ac': 'sitting,writing,raising_hand,standing,turned_around,lie_on_the_desk'}),
                    # person-detection-action-recognition-teacher-0002 is supposed to be provided with -teacher_id, but
                    # this would require providing a gallery file with -fg key. Unless -teacher_id is provided
                    # -teacher_ac is ignored thus run the test just with default actions pretending it's about students
                    TestCase(options={'-m_act': ModelArg('person-detection-action-recognition-teacher-0002')}),
                ],
                [
                    TestCase(options={}),
                    TestCase(options={
                        '-m_lm': ModelArg('landmarks-regression-retail-0009'),
                        '-m_reid': ModelArg('face-recognition-mobilefacenet-arcface'),
                    }),
                ],
            ),
            TestCase(options={'-m_act': ModelArg('person-detection-raisinghand-recognition-0001'), '-a_top': '5'}),
        ],
    )),

    NativeDemo(subdirectory='super_resolution_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'-i': DataDirectoryArg('single-image-super-resolution')}),
        TestCase(options={
            '-m': ModelArg('single-image-super-resolution-1033'),
        }),
    )),

    NativeDemo(subdirectory='text_detection_demo', device_keys=['-d_td', '-d_tr'], test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            **MONITORS,
            '-i': DataPatternArg('text-detection')}),
        single_option_cases('-m_td', ModelArg('text-detection-0003'), ModelArg('text-detection-0004')),
        single_option_cases('-m_tr', None, ModelArg('text-recognition-0012')),
    )),
]

PYTHON_DEMOS = [
    PythonDemo(subdirectory='3d_segmentation_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'-m': ModelArg('brain-tumor-segmentation-0001'),
                          '-o': '.'}),
        single_option_cases('-i', *DATA_SEQUENCES['brain-tumor-nifti']),
    )),

    PythonDemo(subdirectory='action_recognition', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'--no_show': None, **MONITORS, '-i': DataPatternArg('action-recognition')}),
        [
            TestCase(options={
                '-m_en': ModelArg('action-recognition-0001-encoder'),
                '-m_de': ModelArg('action-recognition-0001-decoder'),
            }),
            TestCase(options={
                '-m_en': ModelArg('driver-action-recognition-adas-0002-encoder'),
                '-m_de': ModelArg('driver-action-recognition-adas-0002-decoder'),
            }),
        ],
    )),

    PythonDemo(subdirectory='human_pose_estimation_3d_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'--no_show': None,
                          **MONITORS,
                          '-i': DataPatternArg('human-pose-estimation')}),
        TestCase(options={'-m': ModelArg('human-pose-estimation-3d-0001')}),
    )),

    PythonDemo(subdirectory='image_retrieval_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'--no_show':None,
                          **MONITORS,
                          '-m': ModelArg('image-retrieval-0001')}),
        single_option_cases('-i', *DATA_SEQUENCES['image-retrieval-video']),
        single_option_cases('-g', image_retrieval_arg('gallery.txt')),
    )),

    PythonDemo(subdirectory='instance_segmentation_demo', device_keys=[], test_cases=combine_cases(
        TestCase(options={'--no_show': None,
            **MONITORS,
            '-i': DataPatternArg('instance-segmentation'),
            '--delay': '1',
            '-d': 'CPU',  # GPU is not supported
            '--labels': DemoFileArg('coco_labels.txt')}),
        single_option_cases('-m',
            ModelArg('instance-segmentation-security-0010'),
            ModelArg('instance-segmentation-security-0050'),
            ModelArg('instance-segmentation-security-0083'),
            ModelArg('instance-segmentation-security-1025')),
    )),

    PythonDemo(subdirectory='multi_camera_multi_target_tracking', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'--no_show': None,
            **MONITORS,
            '-i': [DataPatternArg('multi-camera-multi-target-tracking'),
                DataPatternArg('multi-camera-multi-target-tracking/repeated')],
            '-m': ModelArg('person-detection-retail-0013')}),
        single_option_cases('--m_reid',
            ModelArg('person-reidentification-retail-0248'),
            ModelArg('person-reidentification-retail-0265'),
            ModelArg('person-reidentification-retail-0267'),
            ModelArg('person-reidentification-retail-0270')),
    )),

    PythonDemo(subdirectory='object_detection_demo_ssd_async', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'--no_show': None,
            **MONITORS,
            '-i': DataPatternArg('object-detection-demo-ssd-async')}),
        single_option_cases('-m',
            ModelArg('face-detection-adas-0001'),
            ModelArg('face-detection-adas-binary-0001', "FP32-INT1"),
            ModelArg('face-detection-retail-0004'),
            ModelArg('face-detection-retail-0005'),
            ModelArg('face-detection-retail-0044'),
            ModelArg('pedestrian-and-vehicle-detector-adas-0001'),
            ModelArg('pedestrian-detection-adas-0002'),
            ModelArg('pedestrian-detection-adas-binary-0001', "FP32-INT1"),
            ModelArg('person-detection-retail-0013'),
            ModelArg('vehicle-detection-adas-0002'),
            ModelArg('vehicle-detection-adas-binary-0001', "FP32-INT1"),
            ModelArg('vehicle-license-plate-detection-barrier-0106'),
            ModelArg('ssd-resnet34-1200-onnx')),
    )),

    PythonDemo(subdirectory='object_detection_demo_yolov3_async', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'--no_show': None,
            **MONITORS,
            '-i': DataPatternArg('object-detection-demo-ssd-async')}),
        single_option_cases('-m',
            ModelArg('yolo-v1-tiny-tf'),
            ModelArg('yolo-v2-tiny-tf'),
            ModelArg('yolo-v2-tf'),
            ModelArg('yolo-v3-tf'),
            ModelArg('mobilefacedet-v1-mxnet')),
    )),

    PythonDemo(subdirectory='segmentation_demo', device_keys=['-d'], test_cases=combine_cases(
        [
            TestCase(options={
                '-m': ModelArg('road-segmentation-adas-0001'),
                '-i': DATA_SEQUENCES['road-segmentation-adas'],
            }),
            *combine_cases(
                TestCase(options={'-i': DATA_SEQUENCES['semantic-segmentation-adas']}),
                single_option_cases('-m',
                    ModelArg('semantic-segmentation-adas-0001'),
                    ModelArg('deeplabv3'))),
        ],
    )),
]

DEMOS = NATIVE_DEMOS + PYTHON_DEMOS
