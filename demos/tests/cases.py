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

import collections
import itertools
import sys

from args import (
    DataDirectoryArg, DataDirectoryOrigFileNamesArg, DataPatternArg,
    ModelArg, ModelFileArg, OMZ_DIR, TestDataArg, image_net_arg, image_retrieval_arg,
)
from data_sequences import DATA_SEQUENCES

MONITORS = {'-u': 'cdm'}
TestCase = collections.namedtuple('TestCase', ['options'])


class Demo:
    def __init__(self, name, implementation, device_keys=None, test_cases=None):
        self.subdirectory = name + '/' + implementation

        self.device_keys = device_keys

        self.test_cases = test_cases

        self._exec_name = self.subdirectory.replace('/', '_')

    def models_lst_path(self, source_dir):
        return source_dir / self.subdirectory / 'models.lst'

    def device_args(self, device_list):
        if len(self.device_keys) == 0:
            return {'CPU': []}
        return {device: [arg for key in self.device_keys for arg in [key, device]] for device in device_list}


class CppDemo(Demo):
    def __init__(self, name, implementation='cpp', device_keys=None, test_cases=None):
        super().__init__(name, implementation, device_keys, test_cases)

        self._exec_name = self._exec_name.replace('_cpp', '')

    def fixed_args(self, source_dir, build_dir):
        return [str(build_dir / self._exec_name)]


class PythonDemo(Demo):
    def __init__(self, name, implementation='python', device_keys=None, test_cases=None):
        super().__init__(name, implementation, device_keys, test_cases)

        self._exec_name = self._exec_name.replace('_python', '')

    def fixed_args(self, source_dir, build_dir):
        cpu_extension_path = build_dir / 'lib/libcpu_extension.so'

        return [sys.executable, str(source_dir / self.subdirectory / (self._exec_name + '.py')),
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
    CppDemo(name='crossroad_camera_demo',
            device_keys=['-d', '-d_pa', '-d_reid'],
            test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            **MONITORS,
            '-i': DataPatternArg('person-vehicle-bike-detection-crossroad')}),
        TestCase(options={'-m': ModelArg('person-vehicle-bike-detection-crossroad-0078')}),
        single_option_cases('-m_pa', None, ModelArg('person-attributes-recognition-crossroad-0230')),
        single_option_cases('-m_reid',
            None,
            ModelArg('person-reidentification-retail-0277'),
            ModelArg('person-reidentification-retail-0286'),
            ModelArg('person-reidentification-retail-0287'),
            ModelArg('person-reidentification-retail-0288')),
    )),

    CppDemo(name='gaze_estimation_demo',
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

    CppDemo(name='human_pose_estimation_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            **MONITORS,
            '-i': DataPatternArg('human-pose-estimation')}),
        [
            TestCase(options={'-at': 'openpose',
                              '-m': ModelArg('human-pose-estimation-0001')}
            ),
            TestCase(options={'-at': 'higherhrnet',
                              '-m': ModelArg('higher-hrnet-w32-human-pose-estimation')}
            ),
            *combine_cases(
                TestCase(options={'-at': 'ae'}),
                single_option_cases('-m',
                    ModelArg('human-pose-estimation-0005'),
                    ModelArg('human-pose-estimation-0006'),
                    ModelArg('human-pose-estimation-0007')
                )),
        ],
    )),

    CppDemo(name='classification_demo',
            device_keys=['-d'],
            test_cases=combine_cases(
        TestCase(options={
            '-no_show': None,
            '-time': '5',
            '-i': DataDirectoryOrigFileNamesArg('classification'),
            '-labels': str(OMZ_DIR / 'data/dataset_classes/imagenet_2012.txt'),
            '-gt': TestDataArg("ILSVRC2012_img_val/ILSVRC2012_val.txt")}),
        single_option_cases('-m',
            ModelArg('alexnet'),
            ModelArg('densenet-121-tf'),
            ModelArg('densenet-169'),
            ModelArg('mobilenet-v2-pytorch'),
            ModelArg('repvgg-a0'),
            ModelArg('repvgg-b1'),
            ModelArg('repvgg-b3'),
            ModelArg('resnet-50-caffe2')),
    )),

    CppDemo(name='interactive_face_detection_demo',
            device_keys=['-d', '-d_ag', '-d_em', '-d_lm', '-d_hp'],
            test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            **MONITORS,
            '-i': DataPatternArg('375x500')}),
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

    CppDemo(name='interactive_face_detection_demo', implementation='cpp_gapi',
            device_keys=['-d', '-d_ag', '-d_em', '-d_lm', '-d_hp'],
            test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            **MONITORS,
            '-i': DataPatternArg('375x500')}),
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

    CppDemo(name='mask_rcnn_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'-i': DataDirectoryArg('semantic-segmentation-adas')}),
        single_option_cases('-m',
            ModelArg('mask_rcnn_inception_resnet_v2_atrous_coco'),
            ModelArg('mask_rcnn_inception_v2_coco'),
            ModelArg('mask_rcnn_resnet101_atrous_coco'),
            ModelArg('mask_rcnn_resnet50_atrous_coco'))
    )),

    CppDemo(name='multi_channel_face_detection_demo',
            device_keys=['-d'],
            test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            **MONITORS,
            '-i': DATA_SEQUENCES['face-detection-adas']}),
        single_option_cases('-m',
            ModelArg('face-detection-adas-0001'),
            ModelArg('face-detection-retail-0004'),
            ModelArg('face-detection-retail-0005'),
            ModelArg('face-detection-retail-0044')),
    )),

    CppDemo(name='multi_channel_human_pose_estimation_demo', device_keys=['-d'],
            test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            **MONITORS,
            '-i': DATA_SEQUENCES['human-pose-estimation'],
            '-m': ModelArg('human-pose-estimation-0001')}),
    )),

    CppDemo(name='multi_channel_object_detection_demo_yolov3',
            device_keys=['-d'],
            test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            **MONITORS,
             '-i': DataPatternArg('object-detection-demo')}),
        single_option_cases('-m',
            ModelArg('person-vehicle-bike-detection-crossroad-yolov3-1020'),
            ModelArg('yolo-v3-tf'),
            ModelArg('yolo-v3-tiny-tf')),
    )),

    CppDemo(name='object_detection_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'--no_show': None,
            **MONITORS,
            '-i': DataPatternArg('object-detection-demo')}),
        [
            *combine_cases(
                TestCase(options={'-at': 'centernet'}),
                single_option_cases('-m',
                    ModelArg('ctdet_coco_dlav0_384'),
                    ModelArg('ctdet_coco_dlav0_512'))),
            TestCase(options={'-at': 'faceboxes',
                              '-m': ModelArg('faceboxes-pytorch')}
            ),
            *combine_cases(
                TestCase(options={'-at': 'retinaface'}),
                single_option_cases('-m',
                    ModelArg('retinaface-anti-cov'),
                    ModelArg('retinaface-resnet50'),
                    ModelArg('ssh-mxnet'))
            ),
            *combine_cases(
                TestCase(options={'-at': 'ssd'}),
                single_option_cases('-m',
                    ModelArg('efficientdet-d0-tf'),
                    ModelArg('efficientdet-d1-tf'),
                    ModelArg('face-detection-adas-0001'),
                    ModelArg('face-detection-retail-0004'),
                    ModelArg('face-detection-retail-0005'),
                    ModelArg('face-detection-retail-0044'),
                    ModelArg('faster-rcnn-resnet101-coco-sparse-60-0001'),
                    ModelArg('pedestrian-and-vehicle-detector-adas-0001'),
                    ModelArg('pedestrian-detection-adas-0002'),
                    ModelArg('pelee-coco'),
                    ModelArg('person-detection-0200'),
                    ModelArg('person-detection-0201'),
                    ModelArg('person-detection-0202'),
                    ModelArg('person-detection-retail-0013'),
                    ModelArg('person-vehicle-bike-detection-2000'),
                    ModelArg('person-vehicle-bike-detection-2001'),
                    ModelArg('person-vehicle-bike-detection-2002'),
                    #ModelArg('person-vehicle-bike-detection-2003'),
                    #ModelArg('person-vehicle-bike-detection-2004'),
                    ModelArg('product-detection-0001'),
                    ModelArg('rfcn-resnet101-coco-tf'),
                    ModelArg('retinanet-tf'),
                    ModelArg('ssd300'),
                    ModelArg('ssd512'),
                    ModelArg('ssd_mobilenet_v1_coco'),
                    ModelArg('ssd_mobilenet_v1_fpn_coco'),
                    ModelArg('ssd_mobilenet_v2_coco'),
                    ModelArg('ssd_resnet50_v1_fpn_coco'),
                    ModelArg('ssdlite_mobilenet_v2'),
                    ModelArg('vehicle-detection-0200'),
                    ModelArg('vehicle-detection-0201'),
                    ModelArg('vehicle-detection-0201'),
                    ModelArg('vehicle-detection-adas-0002'),
                    ModelArg('vehicle-license-plate-detection-barrier-0106'),
                    ModelArg('vehicle-license-plate-detection-barrier-0123'))),
            *combine_cases(
                TestCase(options={'-at': 'yolo'}),
                single_option_cases('-m',
                    ModelArg('person-vehicle-bike-detection-crossroad-yolov3-1020'),
                    ModelArg('yolo-v3-tf'),
                    ModelArg('yolo-v3-tiny-tf'))),
        ],
    )),

    CppDemo('pedestrian_tracker_demo', device_keys=['-d_det', '-d_reid'], test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            **MONITORS,
            '-i': DataPatternArg('person-detection-retail')}),
        [
            TestCase(options={'-m_det': ModelArg('person-detection-retail-0002')}),
            TestCase(options={'-m_det': ModelArg('person-detection-retail-0013')}),
        ],
        single_option_cases('-m_reid',
            ModelArg('person-reidentification-retail-0277'),
            ModelArg('person-reidentification-retail-0286'),
            ModelArg('person-reidentification-retail-0287'),
            ModelArg('person-reidentification-retail-0288')),
    )),

    CppDemo(name='security_barrier_camera_demo',
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

    CppDemo(name='segmentation_demo', device_keys=['-d'], test_cases=combine_cases(
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
                    ModelArg('fastseg-large'),
                    ModelArg('fastseg-small'),
                    ModelArg('hrnet-v2-c1-segmentation'),
                    ModelArg('deeplabv3'),
                    ModelArg('pspnet-pytorch'))),
        ],
    )),

    CppDemo(name='smart_classroom_demo',
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

    CppDemo(name='super_resolution_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'-i': DataDirectoryArg('single-image-super-resolution')}),
        TestCase(options={
            '-m': ModelArg('single-image-super-resolution-1033'),
        }),
    )),

    CppDemo(name='text_detection_demo', device_keys=['-d_td', '-d_tr'], test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            **MONITORS,
            '-i': DataPatternArg('text-detection')}),
        single_option_cases('-m_td', ModelArg('text-detection-0003'), ModelArg('text-detection-0004')),
        [
            *combine_cases(
                TestCase(options={'-dt': 'ctc'}),
                [
                    *single_option_cases('-m_tr', None, ModelArg('text-recognition-0012')),
                    TestCase(options={'-m_tr': ModelArg('text-recognition-0013'),
                                      '-tr_pt_first': None,
                                      '-tr_o_blb_nm': 'logits'})
                ]),
            TestCase(options={'-m_tr': ModelArg('text-recognition-resnet-fc'),
                              '-tr_pt_first': None,
                              '-dt': 'simple'}),
        ]
    )),
]

PYTHON_DEMOS = [
    PythonDemo(name='speech_recognition_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'-i': TestDataArg('how_are_you_doing.wav')}),
        [
            TestCase(options={'-p': 'mds08x_en',
                              '-m': ModelArg('mozilla-deepspeech-0.8.2'),
                              # run_tests.py puts pre-converted files into dl_dir as
                              # it always runs converter.py without --output_dir
                              '-L': ModelFileArg('mozilla-deepspeech-0.8.2', 'deepspeech-0.8.2-models.kenlm')}),
            TestCase(options={'-p': 'mds06x_en',
                              '-m': ModelArg('mozilla-deepspeech-0.6.1'),
                              # lm.binary is really in dl_dir
                              '-L': ModelFileArg('mozilla-deepspeech-0.6.1', 'lm.binary')}),
            TestCase(options={'-p': 'mds08x_en',  # test online mode
                              '-m': ModelArg('mozilla-deepspeech-0.8.2'),
                              # run_tests.py puts pre-converted files into dl_dir as
                              # it always runs converter.py without --output_dir
                              '-L': ModelFileArg('mozilla-deepspeech-0.8.2', 'deepspeech-0.8.2-models.kenlm'),
                              '--online': None}),
            TestCase(options={'-p': 'mds08x_en',  # test without LM
                              '-m': ModelArg('mozilla-deepspeech-0.8.2')}),
        ],
    )),
]

#DEMOS = NATIVE_DEMOS + PYTHON_DEMOS
DEMOS = PYTHON_DEMOS
