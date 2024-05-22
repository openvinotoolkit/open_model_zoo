# Copyright (c) 2019-2024 Intel Corporation
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
import os
import sys
import typing
from copy import deepcopy

from args import (
    DataDirectoryArg, DataDirectoryOrigFileNamesArg, DataPatternArg,
    ModelArg, ModelFileArg, OMZ_DIR, TestDataArg, image_net_arg, image_retrieval_arg
)
from data_sequences import DATA_SEQUENCES

MONITORS = {'-u': 'cdm'}
UTILIZATION_MONITORS_AND_NO_SHOW_COMMAND_LINE_OPTIONS = {'-u': 'cdm', '--no_show': None}


class TestCase(typing.NamedTuple):
    options: dict
    extra_models: list = []


class Demo:
    IMPLEMENTATION_TYPES = set()

    def __init__(self, name, implementation, model_keys, device_keys, test_cases):
        self.implementation = implementation
        self.model_keys = model_keys
        self.device_keys = device_keys
        self.test_cases = test_cases
        self.subdirectory = name + '/' + implementation
        self._exec_name = self.subdirectory.replace('/', '_')
        self.parser = None
        self.supported_devices = None
        Demo.IMPLEMENTATION_TYPES.add(implementation)

    def models_lst_path(self, source_dir):
        return source_dir / self.subdirectory / 'models.lst'

    def device_args(self, device_list):
        if len(self.device_keys) == 0:
            return {'CPU': []}
        if self.supported_devices:
            device_list = list(set(device_list) & set(self.supported_devices))
        return {device: [arg for key in self.device_keys for arg in [key, device]] for device in device_list}

    def get_models(self, case):
        return ((case.options[key], key) for key in self.model_keys if key in case.options)

    def update_case(self, case, updated_options, with_replacement=False):
        if not updated_options: return
        new_options = case.options.copy()
        for key, value in updated_options.items():
            new_options[key] = value
        new_case = case._replace(options=new_options)
        if with_replacement:
            self.test_cases.remove(case)
        self.test_cases.append(new_case)

    def add_parser(self, parser):
        self.parser = parser(self)
        return self

    def parse_output(self, output, test_case, device):
        if self.parser:
            self.parser(output, test_case, device)

    def update_option(self, updated_options):
        for case in self.test_cases[:]:
            self.update_case(case, updated_options, with_replacement=True)
        return self

    def add_test_cases(self, *new_cases):
        for test_case in new_cases:
            self.test_cases = combine_cases(self.test_cases, test_case)
        return self

    def exclude_models(self, models):
        for case in self.test_cases[:]:
            for model, _ in self.get_models(case):
                if not isinstance(model, ModelArg) or model.name in set(models):
                    self.test_cases.remove(case)
                    continue
        return self

    def only_models(self, models):
        for case in self.test_cases[:]:
            for model, _ in self.get_models(case):
                if not isinstance(model, ModelArg) or model.name not in set(models):
                    self.test_cases.remove(case)
                    continue
        return self

    def only_devices(self, devices):
        self.supported_devices = devices
        return self

    def set_precisions(self, precisions, model_info):
        for case in self.test_cases[:]:
            updated_options = {p: {} for p in precisions}

            for model, key in self.get_models(case):
                if not isinstance(model, ModelArg):
                    continue
                supported_p = list(set(precisions) & set(model_info[model.name]["precisions"]))
                if len(supported_p):
                    model.precision = supported_p[0]
                    for p in supported_p[1:]:
                        updated_options[p][key] = ModelArg(model.name, p)
                else:
                    print("Warning: {} model does not support {} precisions and will not be tested\n".format(
                          model.name, ','.join(precisions)))
                    self.test_cases.remove(case)
                    break

            for p in precisions:
                self.update_case(case, updated_options[p])


class CppDemo(Demo):
    def __init__(self, name, implementation='cpp', model_keys=('-m',), device_keys=('-d',), test_cases=None):
        super().__init__(name, implementation, model_keys, device_keys, test_cases)
        self._exec_name = self._exec_name.replace('_cpp', '')

    def fixed_args(self, source_dir, build_dir):
        return [str(build_dir / self._exec_name)]


class PythonDemo(Demo):
    def __init__(self, name, model_keys=('-m',), device_keys=('-d',), test_cases=None):
        super().__init__(name, 'python', model_keys, device_keys, test_cases)
        self._exec_name = self._exec_name.replace('_python', '')

    def fixed_args(self, source_dir, build_dir):
        if self._exec_name in ('image_retrieval_demo', 'time_series_forecasting_demo', 'object_detection_demo'):
            # sklearn has DeprecationWarning, RuntimeWarning: overflow encountered in exp for yolo-v4-tf
            return [sys.executable, str(source_dir / self.subdirectory / (self._exec_name + '.py'))]
        return [sys.executable, '-W', 'error', str(source_dir / self.subdirectory / (self._exec_name + '.py'))]


def join_cases(*args):
    options = {}
    for case in args: options.update(case.options)
    extra_models = set()
    for case in args: extra_models.update(case.extra_models)
    return TestCase(options=options, extra_models=list(case.extra_models))


def combine_cases(*args):
    return [join_cases(*combination)
        for combination in itertools.product(*[[arg] if isinstance(arg, TestCase) else arg for arg in args])]


def single_option_cases(key, *args):
    return [TestCase(options={} if arg is None else {key: arg}) for arg in args]



DEMOS = [
    # CppDemo(name='background_subtraction_demo', device_keys=['-d'], implementation='cpp_gapi', test_cases=combine_cases(
    #     TestCase(options={'--no_show': None, '-at': 'maskrcnn',
    #         **MONITORS,
    #         '-i': DataPatternArg('coco128-subset-480x640x3'),
    #     }),
    #     single_option_cases('-m',
    #         ModelArg('instance-segmentation-person-0007'),
    #         ModelArg('instance-segmentation-security-0091')),
    # )),

    CppDemo('classification_benchmark_demo', 'cpp_gapi', test_cases=combine_cases(
        single_option_cases(
            '-m',
            ModelArg('convnext-tiny'),
            # TODO: enable after https://github.com/TolyaTalamanov fixes G-API
            # ModelArg('densenet-121-tf'),
            ModelArg('dla-34'),
            # ModelArg('efficientnet-b0'),
            ModelArg('efficientnet-b0-pytorch'),
            ModelArg('efficientnet-v2-b0'),
            ModelArg('efficientnet-v2-s'),
            # ModelArg('googlenet-v1-tf'),
            # ModelArg('googlenet-v2-tf'),
            # ModelArg('googlenet-v3'),
            ModelArg('googlenet-v3-pytorch'),
            # ModelArg('googlenet-v4-tf'),
            ModelArg('hbonet-0.25'),
            ModelArg('hbonet-1.0'),
            # ModelArg('inception-resnet-v2-tf'),
            ModelArg('levit-128s'),
            # ModelArg('mixnet-l'),
            # ModelArg('mobilenet-v1-0.25-128'),
            # ModelArg('mobilenet-v1-1.0-224-tf'),
            # ModelArg('mobilenet-v2-1.0-224'),
            # ModelArg('mobilenet-v2-1.4-224'),
            ModelArg('mobilenet-v2-pytorch'),
            # ModelArg('mobilenet-v3-large-1.0-224-tf'),
            # ModelArg('mobilenet-v3-small-1.0-224-tf'),
            ModelArg('nfnet-f0'),
            ModelArg('regnetx-3.2gf'),
            ModelArg('repvgg-a0'),
            ModelArg('repvgg-b1'),
            ModelArg('repvgg-b3'),
            ModelArg('resnest-50-pytorch'),
            ModelArg('resnet-18-pytorch'),
            ModelArg('resnet-34-pytorch'),
            ModelArg('resnet-50-pytorch'),
            # ModelArg('resnet-50-tf'),
            ModelArg('resnet18-xnor-binary-onnx-0001'),
            ModelArg('resnet50-binary-0001'),
            ModelArg('rexnet-v1-x1.0'),
            ModelArg('shufflenet-v2-x1.0'),
            ModelArg('swin-tiny-patch4-window7-224'),
            ModelArg('t2t-vit-14'),
        ),
        TestCase({
            '-time': '5',
            '-i': TestDataArg('coco128/images/train2017/'),
            '-labels': str(OMZ_DIR / 'data/dataset_classes/imagenet_2012.txt'),
            **UTILIZATION_MONITORS_AND_NO_SHOW_COMMAND_LINE_OPTIONS
        })
    )),

    CppDemo(name='gaze_estimation_demo', implementation='cpp_gapi',
            model_keys=['-m', '-m_fd', '-m_hp', '-m_lm', '-m_es'],
            device_keys=['-d', '-d_fd', '-d_hp', '-d_lm', '-d_es'],
            test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            **MONITORS}),
        TestCase(options={
            '-m': ModelArg('gaze-estimation-adas-0002'),
            '-m_hp': ModelArg('head-pose-estimation-adas-0001'),
            '-m_lm': ModelArg('facial-landmarks-35-adas-0002'),
            '-m_es': ModelArg('open-closed-eye-0001'),
        }),
        single_option_cases(
            '-m_fd',
            ModelArg('face-detection-adas-0001'),
            ModelArg('face-detection-retail-0004')),
        single_option_cases(
            '-i',
            str('video.mp4'),
            DataPatternArg('coco128-every-480x640x3')),
    )),

    # TODO: https://github.com/DariaMityagina is to fix the demo
    # CppDemo(name='gesture_recognition_demo', implementation='cpp_gapi',
    #         model_keys=['-m_a', '-m_d'],
    #         device_keys=['-d_a', '-d_d'],
    #         test_cases=combine_cases(
    #     TestCase(options={'--no_show': None,
    #                       '-i': DataPatternArg('coco128-every-480x640x3'),
    #                       '-m_d': ModelArg('person-detection-asl-0001')}),
    #     [
    #         TestCase(options={'-m_a': ModelArg('asl-recognition-0004'), '-c': str(OMZ_DIR / 'data/dataset_classes/msasl100.json')}),
    #         TestCase(options={'-m_a': ModelArg('common-sign-language-0001'),
    #                           '-c': str(OMZ_DIR / 'data/dataset_classes/jester27.json')}),
    #         TestCase(options={'-m_a': ModelArg('common-sign-language-0002'),
    #                           '-c': str(OMZ_DIR / 'data/dataset_classes/common_sign_language12.json')}),
    #     ],
    # )),

    CppDemo(
        'interactive_face_detection_demo', 'cpp_gapi',
        ('-m', '--mag', '--mem', '--mlm', '--mhp', '--mam'), ('-d', '--dag', '--dem', '--dlm', '--dhp', '--dam'),
        combine_cases(
            [
                TestCase({
                    '-m': ModelArg('face-detection-retail-0004'),
                    '--mag': ModelArg('age-gender-recognition-retail-0013'),
                    '--mam': ModelArg('anti-spoof-mn3'),
                    '--mem': ModelArg('emotions-recognition-retail-0003'),
                    '--mhp': ModelArg('head-pose-estimation-adas-0001'),
                    '--mlm': ModelArg('facial-landmarks-35-adas-0002'),
                }),
                TestCase({'-m': ModelArg('face-detection-adas-0001')})
            ],
            TestCase({
                '-i': DataPatternArg('coco128-every-480x640x3'),
                **MONITORS,
                '--noshow': None
            }),
        )
    ),

    CppDemo(name='smart_classroom_demo', implementation='cpp_gapi',
            model_keys=['-m_act', '-m_fd', '-m_lm', '-m_reid'],
            device_keys=['-d_act', '-d_fd', '-d_lm', '-d_reid'],
            test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            **MONITORS,
            '-i': DataPatternArg('coco128-subset-480x640x3'),
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
                        '-m_reid': ModelArg('face-recognition-resnet100-arcface-onnx'),
                    }),
                ],
            ),
            TestCase(options={'-m_act': ModelArg('person-detection-raisinghand-recognition-0001'), '-a_top': '5'}),
        ],
    )),

    CppDemo(name='classification_benchmark_demo',
            device_keys=['-d'],
            test_cases=combine_cases(
        TestCase(options={
            '-no_show': None,
            '-time': '5',
            '-i': DataDirectoryOrigFileNamesArg('classification'),
            '-labels': str(OMZ_DIR / 'data/dataset_classes/imagenet_2012.txt'),
            '-gt': TestDataArg("ILSVRC2012_img_val/ILSVRC2012_val.txt")}),
        single_option_cases(
            '-m',
            ModelArg('convnext-tiny'),
            ModelArg('densenet-121-tf'),
            ModelArg('dla-34'),
            ModelArg('efficientnet-b0'),
            ModelArg('efficientnet-b0-pytorch'),
            ModelArg('efficientnet-v2-b0'),
            ModelArg('efficientnet-v2-s'),
            ModelArg('googlenet-v1-tf'),
            ModelArg('googlenet-v2-tf'),
            ModelArg('googlenet-v3'),
            ModelArg('googlenet-v3-pytorch'),
            ModelArg('googlenet-v4-tf'),
            ModelArg('hbonet-0.25'),
            ModelArg('hbonet-1.0'),
            ModelArg('inception-resnet-v2-tf'),
            ModelArg('levit-128s'),
            ModelArg('mixnet-l'),
            ModelArg('mobilenet-v1-0.25-128'),
            ModelArg('mobilenet-v1-1.0-224-tf'),
            ModelArg('mobilenet-v2-1.0-224'),
            ModelArg('mobilenet-v2-1.4-224'),
            ModelArg('mobilenet-v2-pytorch'),
            ModelArg('mobilenet-v3-large-1.0-224-tf'),
            ModelArg('mobilenet-v3-small-1.0-224-tf'),
            ModelArg('nfnet-f0'),
            ModelArg('regnetx-3.2gf'),
            ModelArg('repvgg-a0'),
            ModelArg('repvgg-b1'),
            ModelArg('repvgg-b3'),
            ModelArg('resnest-50-pytorch'),
            ModelArg('resnet-18-pytorch'),
            ModelArg('resnet-34-pytorch'),
            ModelArg('resnet-50-pytorch'),
            ModelArg('resnet-50-tf'),
            ModelArg('resnet18-xnor-binary-onnx-0001'),
            ModelArg('resnet50-binary-0001'),
            ModelArg('rexnet-v1-x1.0'),
            ModelArg('shufflenet-v2-x1.0'),
            ModelArg('t2t-vit-14'),
    ))),

    CppDemo(name='crossroad_camera_demo',
            model_keys=['-m', '-m_pa', '-m_reid'],
            device_keys=['-d', '-d_pa', '-d_reid'],
            test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            **MONITORS,
            '-i': DataPatternArg('person-vehicle-bike-detection-crossroad')}),
        single_option_cases(
            '-m',
            ModelArg('person-vehicle-bike-detection-crossroad-0078'),
            ModelArg('person-vehicle-bike-detection-crossroad-1016'),
        ),
        single_option_cases(
            '-m_pa',
            None,
            ModelArg('person-attributes-recognition-crossroad-0230'),
            ModelArg('person-attributes-recognition-crossroad-0234'),
            ModelArg('person-attributes-recognition-crossroad-0238'),
        ),
        single_option_cases('-m_reid',
            None,
            ModelArg('person-reidentification-retail-0277'),
            ModelArg('person-reidentification-retail-0286'),
            ModelArg('person-reidentification-retail-0287'),
            ModelArg('person-reidentification-retail-0288')
        ),
    )),

    CppDemo(name='gaze_estimation_demo',
            model_keys=['-m', '-m_fd', '-m_hp', '-m_lm', '-m_es'],
            device_keys=['-d', '-d_fd', '-d_hp', '-d_lm', '-d_es'],
            test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            **MONITORS,
            '-i': DataPatternArg('gaze-estimation-adas')}),
        TestCase(options={
            '-m': ModelArg('gaze-estimation-adas-0002'),
            '-m_hp': ModelArg('head-pose-estimation-adas-0001'),
            '-m_es': ModelArg('open-closed-eye-0001'),
        }),
        [
        *combine_cases(
            single_option_cases('-m_lm',
                ModelArg('facial-landmarks-35-adas-0002'),
                ModelArg('facial-landmarks-98-detection-0001'),
        )),
        ],
        [
        *combine_cases(
            single_option_cases('-m_fd',
                ModelArg('face-detection-adas-0001'),
                ModelArg('face-detection-retail-0004')
        )),
        ],
    )),

    CppDemo(name='human_pose_estimation_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            **MONITORS,
            '-i': DataPatternArg('human-pose-estimation')}),
        [
            TestCase(options={'-at': 'openpose',
                              '-m': ModelArg('human-pose-estimation-0001')}
            ),
            TestCase({'-at': 'higherhrnet', '-m': ModelArg('higher-hrnet-w32-human-pose-estimation')}),
            *combine_cases(
                TestCase(options={'-at': 'ae'}),
                single_option_cases('-m',
                    ModelArg('human-pose-estimation-0005'),
                    ModelArg('human-pose-estimation-0006'),
                    ModelArg('human-pose-estimation-0007')
                )),
        ],
    )),

    CppDemo(name='image_processing_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'--no_show': None,
            **MONITORS,
            '-i': DataDirectoryArg('single-image-super-resolution')}),
        [
            *combine_cases(
                TestCase(options={'-at': 'sr'}),
                single_option_cases('-m',
                    ModelArg('single-image-super-resolution-1032'),
                    ModelArg('single-image-super-resolution-1033'),
                    ModelArg('text-image-super-resolution-0001'))
            ),
            TestCase({'-at': 'jr', '-m': ModelArg('fbcnn')}),
            TestCase({'-at': 'style', '-m': ModelArg('fast-neural-style-mosaic-onnx')}),
        ]
    )),

    CppDemo(name='interactive_face_detection_demo',
            model_keys=['-m', '--mag', '--mem', '--mlm', '--mhp', '--mam'],
            device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'--noshow': None,
            **MONITORS,
            '-i': DataPatternArg('375x500')}),
        [
            TestCase(options={
                '-m': ModelArg('face-detection-retail-0004'),
                '--mag': ModelArg('age-gender-recognition-retail-0013'),
                '--mam': ModelArg('anti-spoof-mn3'),
                '--mem': ModelArg('emotions-recognition-retail-0003'),
                '--mhp': ModelArg('head-pose-estimation-adas-0001'),
                '--mlm': ModelArg('facial-landmarks-35-adas-0002'),
            }),
            TestCase(options={'-m': ModelArg('face-detection-adas-0001')})
        ]
    )),

    CppDemo(name='mask_rcnn_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'-i': DataDirectoryArg('instance-segmentaion-mask-rcnn')}),
        single_option_cases('-m',
            ModelArg('mask_rcnn_inception_resnet_v2_atrous_coco'),
            ModelArg('mask_rcnn_resnet50_atrous_coco'))
    )),

    CppDemo(name='multi_channel_face_detection_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            **MONITORS,
            '-i': DATA_SEQUENCES['face-detection-adas']}),
        [
            TestCase(options={'-m':  ModelArg('face-detection-adas-0001')}),
            TestCase(options={'-m':  ModelArg('face-detection-retail-0004'), '-bs': '2',
                '-show_stats': '', '-n_iqs': '1', '-duplicate_num': '2'}),
            TestCase(options={'-m':  ModelArg('face-detection-retail-0005'), '-bs': '3',
                '-n_iqs': '999'}),
            TestCase(options={'-m':  ModelArg('face-detection-adas-0001'), '-bs': '4',
                '-show_stats': '', '-duplicate_num': '3', '-real_input_fps': ''})
        ]
    )),

    CppDemo(name='multi_channel_human_pose_estimation_demo', device_keys=['-d'],
        test_cases=[TestCase(options={'-no_show': None,
            **MONITORS,
            '-i': DATA_SEQUENCES['human-pose-estimation'],
            '-m': ModelArg('human-pose-estimation-0001')}),
    ]),

    CppDemo(name='multi_channel_object_detection_demo_yolov3', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            **MONITORS,
             '-i': DataPatternArg('object-detection-demo')}),
        [
            TestCase(options={'-m':  ModelArg('person-vehicle-bike-detection-crossroad-yolov3-1020')}),
            TestCase(options={'-m':  ModelArg('yolo-v3-tf'), '-duplicate_num': '2',
                '-n_iqs': '20', '-fps_sp': '1', '-n_sp': '1', '-show_stats': '', '-real_input_fps': ''}),
            TestCase(options={'-m':  ModelArg('yolo-v3-tiny-tf'), '-duplicate_num': '3',
                '-n_iqs': '9999', '-fps_sp': '50', '-n_sp': '30'})
        ]
    )),

    CppDemo(name='noise_suppression_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'-i': TestDataArg('how_are_you_doing.wav')}),
        single_option_cases('-m',
            ModelArg('noise-suppression-denseunet-ll-0001'),
            ModelArg('noise-suppression-poconetlike-0001')),
    )),

    CppDemo(name='object_detection_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'--no_show': None,
            **MONITORS,
            '-i': DataPatternArg('object-detection-demo')}),
        [
            *combine_cases(
                TestCase(options={'-at': 'centernet'}),
                [
                    *single_option_cases('-m',
                        ModelArg('ctdet_coco_dlav0_512'),
                    ),
                    *combine_cases(
                        TestCase(options={
                            '-mean_values': "104.04 113.985 119.85",
                            '-scale_values': "73.695 69.87 70.89",
                        }),
                        single_option_cases('-m',
                            ModelFileArg('ctdet_coco_dlav0_512', 'ctdet_coco_dlav0_512.onnx'),
                        ),
                    ),
                ]
            ),
            *combine_cases(
                TestCase(options={'-at': 'faceboxes'}),
                [
                    TestCase(options={'-m': ModelArg('faceboxes-pytorch')}),
                    TestCase(options={'-m': ModelFileArg('faceboxes-pytorch', 'faceboxes-pytorch.onnx'),
                                      '-mean_values': "104.0 117.0 123.0"}),
                ]
            ),
            *combine_cases(
                TestCase(options={'-at': 'retinaface-pytorch'}),
                [
                    TestCase(options={'-m': ModelArg('retinaface-resnet50-pytorch')}),
                    TestCase({
                        '-m': ModelFileArg('retinaface-resnet50-pytorch', 'retinaface-resnet50-pytorch.onnx'),
                        '-mean_values': "104.0 117.0 123.0",
                    }),
                ]
            ),
            *combine_cases(
                TestCase(options={'-at': 'ssd'}),
                [
                    *single_option_cases('-m',
                        ModelArg('efficientdet-d0-tf'),
                        ModelArg('efficientdet-d1-tf'),
                        ModelArg('face-detection-0200'),
                        ModelArg('face-detection-0202'),
                        ModelArg('face-detection-0204'),
                        ModelArg('face-detection-0205'),
                        ModelArg('face-detection-0206'),
                        ModelArg('face-detection-adas-0001'),
                        ModelArg('face-detection-retail-0004'),
                        ModelArg('face-detection-retail-0005'),
                        ModelArg('faster_rcnn_inception_resnet_v2_atrous_coco'),
                        ModelArg('faster_rcnn_resnet50_coco'),
                        ModelArg('faster-rcnn-resnet101-coco-sparse-60-0001'),
                        ModelArg('pedestrian-and-vehicle-detector-adas-0001'),
                        ModelArg('pedestrian-detection-adas-0002'),
                        ModelArg('person-detection-0200'),
                        ModelArg('person-detection-0201'),
                        ModelArg('person-detection-0202'),
                        ModelArg('person-detection-0203'),
                        ModelArg('person-detection-0301'),
                        ModelArg('person-detection-0302'),
                        ModelArg('person-detection-0303'),
                        ModelArg('person-detection-retail-0013'),
                        ModelArg('person-vehicle-bike-detection-2000'),
                        ModelArg('person-vehicle-bike-detection-2001'),
                        ModelArg('person-vehicle-bike-detection-2002'),
                        ModelArg('person-vehicle-bike-detection-2003'),
                        ModelArg('person-vehicle-bike-detection-2004'),
                        ModelArg('product-detection-0001'),
                        ModelArg('rfcn-resnet101-coco-tf'),
                        ModelArg('retinanet-tf'),
                        ModelArg('ssd_mobilenet_v1_coco'),
                        ModelArg('ssd_mobilenet_v1_fpn_coco'),
                        ModelArg('ssdlite_mobilenet_v2'),
                        ModelArg('vehicle-detection-0200'),
                        ModelArg('vehicle-detection-0201'),
                        ModelArg('vehicle-detection-0202'),
                        ModelArg('vehicle-detection-adas-0002'),
                        ModelArg('vehicle-license-plate-detection-barrier-0106'),
                        ModelArg('vehicle-license-plate-detection-barrier-0123')),
                    TestCase(options={'-m': ModelFileArg('ssd-resnet34-1200-onnx', 'resnet34-ssd1200.onnx'),
                                        '-reverse_input_channels': None,
                                        '-mean_values': "123.675 116.28 103.53",
                                        '-scale_values': "58.395 57.12 57.375"}),
                ]
            ),
            *combine_cases(
                TestCase(options={'-at': 'yolo'}),
                single_option_cases('-m',
                    ModelArg('mobilenet-yolo-v4-syg'),
                    ModelArg('person-vehicle-bike-detection-crossroad-yolov3-1020'),
                    ModelArg('yolo-v1-tiny-tf'),
                    ModelArg('yolo-v2-ava-0001'),
                    ModelArg('yolo-v2-ava-sparse-35-0001'),
                    ModelArg('yolo-v2-ava-sparse-70-0001'),
                    ModelArg('yolo-v2-tiny-ava-0001'),
                    ModelArg('yolo-v2-tiny-ava-sparse-30-0001'),
                    ModelArg('yolo-v2-tiny-ava-sparse-60-0001'),
                    ModelArg('yolo-v2-tiny-vehicle-detection-0001'),
                    ModelArg('yolo-v2-tf'),
                    ModelArg('yolo-v2-tiny-tf'),
                    ModelArg('yolo-v3-tf'),
                    ModelArg('yolo-v3-tiny-tf'),
                    ModelArg('yolo-v4-tf'),
                    ModelArg('yolo-v4-tiny-tf'))),
        ],
    )),

    CppDemo(name='pedestrian_tracker_demo',
            model_keys=['-m_det', '-m_reid'],
            device_keys=['-d_det', '-d_reid'],
            test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            **MONITORS,
            '-i': DataPatternArg('person-detection-retail')}),
        [
            *combine_cases(
                TestCase(options={'-at': 'ssd'}),
                single_option_cases('-m_det',
                    ModelArg('person-detection-retail-0002'),
                    ModelArg('person-detection-retail-0013')),
            ),
            TestCase(options={'-person_label': '0', '-at': 'yolo', '-m_det': ModelArg('yolo-v3-tf')}),
            TestCase(options={'-person_label': '0', '-at': 'centernet', '-m_det': ModelArg('ctdet_coco_dlav0_512')}),
            TestCase(options={'-person_label': '1', '-at': 'ssd', '-m_det': ModelArg('retinanet-tf')}),
        ],
        single_option_cases('-m_reid',
            ModelArg('person-reidentification-retail-0277'),
            ModelArg('person-reidentification-retail-0286'),
            ModelArg('person-reidentification-retail-0287'),
            ModelArg('person-reidentification-retail-0288')),
    )),

    CppDemo(name='security_barrier_camera_demo',
            model_keys=['-m', '-m_lpr', '-m_va'],
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
        single_option_cases(
            '-m_va',
            None,
            ModelArg('vehicle-attributes-recognition-barrier-0039'),
            ModelArg('vehicle-attributes-recognition-barrier-0042'),
        ),
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
                    ModelArg('pspnet-pytorch'),
                    ModelArg('drn-d-38'),
                    ModelArg('erfnet'),
                )),
        ],
    )),

    CppDemo(name='smart_classroom_demo',
            model_keys=['-m_act', '-m_fd', '-m_lm', '-m_reid'],
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
                        '-m_reid': ModelArg('face-recognition-resnet100-arcface-onnx'),
                    }),
                ],
            ),
            TestCase(options={'-m_act': ModelArg('person-detection-raisinghand-recognition-0001'), '-a_top': '5'}),
        ],
    )),

    CppDemo(name='social_distance_demo', device_keys=['-d_det', '-d_reid'],
            model_keys=['-m_det', '-m_reid'], test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            **MONITORS,
            '-i': DataDirectoryArg('person-detection-retail')}),
        single_option_cases('-m_det',
            ModelArg('person-detection-0200'),
            ModelArg('person-detection-0201'),
            ModelArg('person-detection-0202'),
            ModelArg('person-detection-retail-0013')),
        single_option_cases('-m_reid',
            ModelArg('person-reidentification-retail-0277'),
            ModelArg('person-reidentification-retail-0286'),
            ModelArg('person-reidentification-retail-0287'),
            ModelArg('person-reidentification-retail-0288'),
        ),
    )),

    CppDemo(name='text_detection_demo', model_keys=['-m_td', '-m_tr'], device_keys=['-d_td', '-d_tr'],
            test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            **MONITORS,
            '-i': DataPatternArg('text-detection')}),
        single_option_cases('-m_td',
            ModelArg('text-detection-0003'),
            ModelArg('text-detection-0004'),
            ModelArg('horizontal-text-detection-0001')),
        [
            *combine_cases(
                TestCase(options={'-dt': 'ctc'}),
                [
                    *single_option_cases('-m_tr', None, ModelArg('text-recognition-0012')),
                    TestCase(options={'-m_tr': ModelArg('text-recognition-0014'),
                                      '-tr_pt_first': None,
                                      '-tr_o_blb_nm': 'logits'}),
                    TestCase({'-m_tr': ModelArg('handwritten-score-recognition-0003'), '-m_tr_ss': '0123456789._'})
                ]),
            *combine_cases(
                TestCase(options={'-dt': 'simple'}),
                [
                    TestCase(options={'-m_tr': ModelArg('text-recognition-0015-encoder'),
                                      '-tr_pt_first': None,
                                      '-tr_o_blb_nm': 'logits',
                                      '-m_tr_ss': '?0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'},
                             extra_models=[ModelArg('text-recognition-0015-decoder')]),
                    TestCase(options={'-m_tr': ModelArg('text-recognition-0016-encoder'),
                                       '-tr_pt_first': None,
                                       '-tr_o_blb_nm': 'logits',
                                       '-m_tr_ss': '?0123456789abcdefghijklmnopqrstuvwxyz'},
                              extra_models=[ModelArg('text-recognition-0016-decoder')]),
                    TestCase(options={'-m_tr': ModelArg('text-recognition-resnet-fc'),
                                      '-tr_pt_first': None}),
                    TestCase(options={'-m_tr': ModelArg('vitstr-small-patch16-224'),
                                      '-tr_pt_first': None,
                                      '-m_tr_ss': str(OMZ_DIR / 'models/public/vitstr-small-patch16-224/vocab.txt'),
                                      '-start_index': '1',
                                      '-pad': " "}),
                ]),
        ]
    )),

    PythonDemo(name='3d_segmentation_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'-m': ModelArg('brain-tumor-segmentation-0002'),
                          '-o': '.',
                          '-ms': '1,2,3,0'}),
        single_option_cases('-i', *DATA_SEQUENCES['brain-tumor-nifti']),
    )),

    PythonDemo(name='action_recognition_demo', device_keys=['-d'],
               model_keys=['-m_en', '-m_de'], test_cases=combine_cases(
        TestCase(options={'--no_show': None, **MONITORS, '-i': DataPatternArg('action-recognition')}),
        [
            TestCase(options={'--architecture_type': 'i3d-rgb',
                              '-m_en': ModelArg('i3d-rgb-tf')}
            ),
            TestCase({'--architecture_type': 'en-mean', '-m_en': ModelArg('weld-porosity-detection-0001')}),
            *combine_cases(
               TestCase(options={'--architecture_type': 'en-de'}),
               [
                   TestCase(options={
                       '-m_en': ModelArg('action-recognition-0001-encoder'),
                       '-m_de': ModelArg('action-recognition-0001-decoder'),
                   }),
                   TestCase(options={
                       '-m_en': ModelArg('driver-action-recognition-adas-0002-encoder'),
                       '-m_de': ModelArg('driver-action-recognition-adas-0002-decoder'),
                   }),
               ]
            ),
        ],
    )),

    PythonDemo(name='background_subtraction_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'--no_show': None,
            **MONITORS,
            '-i': DataPatternArg('instance-segmentation'),
            '--background': DataPatternArg('instance-segmentation'),
        }),
        single_option_cases('-m',
            ModelArg('instance-segmentation-person-0007'),
            ModelArg('robust-video-matting-mobilenetv3'),
            ModelArg('background-matting-mobilenetv2'),
            ModelArg('yolact-resnet50-fpn-pytorch'),
            ModelArg('modnet-photographic-portrait-matting'),
            ModelArg('modnet-webcam-portrait-matting')),
    )),

    PythonDemo(name='bert_question_answering_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'-i': 'https://en.wikipedia.org/wiki/OpenVINO',
                          '--questions': ['What frameworks does OpenVINO support?', 'Who are developers?']}),
        [
            TestCase(options={
                '-m': ModelArg('bert-small-uncased-whole-word-masking-squad-0001'),
                '--input_names': 'input_ids,attention_mask,token_type_ids',
                '--output_names': 'output_s,output_e',
                '--vocab': ModelFileArg('bert-small-uncased-whole-word-masking-squad-0001', 'vocab.txt'),
            }),
            TestCase(options={
                '-m': ModelArg('bert-small-uncased-whole-word-masking-squad-0002'),
                '--input_names': 'input_ids,attention_mask,token_type_ids,position_ids',
                '--output_names': 'output_s,output_e',
                '--vocab': ModelFileArg('bert-small-uncased-whole-word-masking-squad-0002', 'vocab.txt'),
            }),
            TestCase(options={
                '-m': ModelArg('bert-small-uncased-whole-word-masking-squad-int8-0002'),
                '--input_names': 'input_ids,attention_mask,token_type_ids,position_ids',
                '--output_names': 'output_s,output_e',
                '--vocab': ModelFileArg('bert-small-uncased-whole-word-masking-squad-int8-0002', 'vocab.txt'),
            }),
            TestCase(options={
                '-m': ModelArg('bert-large-uncased-whole-word-masking-squad-0001'),
                '--input_names': 'input_ids,attention_mask,token_type_ids',
                '--output_names': 'output_s,output_e',
                '--vocab': ModelFileArg('bert-large-uncased-whole-word-masking-squad-0001', 'vocab.txt'),
            }),
            TestCase(options={
                '-m': ModelArg('bert-large-uncased-whole-word-masking-squad-int8-0001'),
                '--input_names': 'input_ids,attention_mask,token_type_ids',
                '--output_names': 'output_s,output_e',
                '--vocab': ModelFileArg('bert-large-uncased-whole-word-masking-squad-int8-0001', 'vocab.txt')
            }),
        ]
    )),

    PythonDemo(name='bert_question_answering_embedding_demo', device_keys=['-d'],
               model_keys=['-m_emb', '-m_qa'], test_cases=combine_cases(
        TestCase(options={'-i': 'https://en.wikipedia.org/wiki/OpenVINO',
                          '--questions': ['What frameworks does OpenVINO support?', 'Who are developers?']}),
        [
            TestCase(options={
                '-m_emb': ModelArg('bert-large-uncased-whole-word-masking-squad-emb-0001'),
                '--input_names_emb': 'input_ids,attention_mask,token_type_ids,position_ids',
                '--vocab': ModelFileArg('bert-large-uncased-whole-word-masking-squad-emb-0001', 'vocab.txt'),
                '-m_qa': ModelArg('bert-small-uncased-whole-word-masking-squad-0001'),
                '--input_names_qa': 'input_ids,attention_mask,token_type_ids',
                '--output_names_qa': 'output_s,output_e',
            }),
            TestCase(options={
                '-m_emb': ModelArg('bert-large-uncased-whole-word-masking-squad-emb-0001'),
                '--input_names_emb': 'input_ids,attention_mask,token_type_ids,position_ids',
                '--vocab': ModelFileArg('bert-large-uncased-whole-word-masking-squad-emb-0001', 'vocab.txt'),
            }),
            TestCase(options={
                '-m_emb': ModelArg('bert-small-uncased-whole-word-masking-squad-emb-int8-0001'),
                '--input_names_emb': 'input_ids,attention_mask,token_type_ids,position_ids',
                '--vocab': ModelFileArg('bert-small-uncased-whole-word-masking-squad-emb-int8-0001', 'vocab.txt'),
            }),
        ]
    )),

    PythonDemo(name='bert_named_entity_recognition_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={
            '-nireq': '1', # launch demo in synchronous mode
            '-i': 'https://en.wikipedia.org/wiki/OpenVINO',
            '-m': ModelArg('bert-base-ner'),
            '-v': ModelFileArg('bert-base-ner', 'bert-base-ner/vocab.txt')
        }),
    )),

    PythonDemo(name='classification_demo',
            device_keys=['-d'],
            test_cases=combine_cases(
        TestCase(options={
            '--no_show': None,
            '-i': DataDirectoryOrigFileNamesArg('classification'),
            '--labels': str(OMZ_DIR / 'data/dataset_classes/imagenet_2012.txt')}),
        [
            *single_option_cases(
                '-m',
                ModelArg('convnext-tiny'),
                ModelArg('densenet-121-tf'),
                ModelArg('dla-34'),
                ModelArg('efficientnet-b0'),
                ModelArg('efficientnet-b0-pytorch'),
                ModelArg('efficientnet-v2-b0'),
                ModelArg('efficientnet-v2-s'),
                ModelArg('googlenet-v1-tf'),
                ModelArg('googlenet-v2-tf'),
                ModelArg('googlenet-v3'),
                ModelArg('googlenet-v3-pytorch'),
                ModelArg('googlenet-v4-tf'),
                ModelArg('hbonet-0.25'),
                ModelArg('hbonet-1.0'),
                ModelArg('inception-resnet-v2-tf'),
                ModelArg('levit-128s'),
                ModelArg('mixnet-l'),
                ModelArg('mobilenet-v1-0.25-128'),
                ModelArg('mobilenet-v1-1.0-224-tf'),
                ModelArg('mobilenet-v2-1.0-224'),
                ModelArg('mobilenet-v2-1.4-224'),
                ModelArg('mobilenet-v2-pytorch'),
                ModelArg('mobilenet-v3-large-1.0-224-tf'),
                ModelArg('mobilenet-v3-small-1.0-224-tf'),
                ModelArg('nfnet-f0'),
                ModelArg('regnetx-3.2gf'),
                ModelArg('repvgg-a0'),
                ModelArg('repvgg-b1'),
                ModelArg('repvgg-b3'),
                ModelArg('resnest-50-pytorch'),
                ModelArg('resnet-18-pytorch'),
                ModelArg('resnet-34-pytorch'),
                ModelArg('resnet-50-pytorch'),
                ModelArg('resnet-50-tf'),
                ModelArg('resnet18-xnor-binary-onnx-0001'),
                ModelArg('resnet50-binary-0001'),
                ModelArg('rexnet-v1-x1.0'),
                ModelArg('shufflenet-v2-x1.0'),
                ModelArg('swin-tiny-patch4-window7-224'),
                ModelArg('t2t-vit-14'),
            ),

            TestCase(options={'-m': ModelFileArg('efficientnet-b0-pytorch', 'efficientnet-b0.onnx'),
                        '--reverse_input_channels': None,
                        '--mean_values': ['123.675', '116.28', '103.53'],
                        '--scale_values': ['58.395', '57.12', '57.375']}),
        ]
    )),

    PythonDemo('colorization_demo', test_cases=combine_cases(
        single_option_cases(
            '-m',
            ModelArg('colorization-v2'),
            ModelArg('colorization-siggraph'),
            ModelFileArg('colorization-siggraph', 'colorization-siggraph.onnx'),
        ),
        TestCase({
            '-i': DataPatternArg('classification'),
            '-m': ModelArg('colorization-v2'),
            **UTILIZATION_MONITORS_AND_NO_SHOW_COMMAND_LINE_OPTIONS
        })
    )),

    PythonDemo(name='face_recognition_demo', device_keys=['-d_fd', '-d_lm', '-d_reid'],
               model_keys=['-m_fd', '-m_lm', '-m_reid'], test_cases=combine_cases(
        TestCase(options={'--no_show': None,
                          **MONITORS,
                          '-i': DataPatternArg('face-detection-adas'),
                          '-fg': DataDirectoryArg('face-recognition-gallery')
                          }),
        single_option_cases('-m_fd',
            ModelArg('face-detection-adas-0001'),
            ModelArg('face-detection-retail-0004'),
            ModelArg('face-detection-retail-0005')),
        single_option_cases('-m_lm', ModelArg('landmarks-regression-retail-0009')),
        single_option_cases('-m_reid',
            ModelArg('face-reidentification-retail-0095'),
            ModelArg('face-recognition-resnet100-arcface-onnx'),
            ModelArg('facenet-20180408-102900')),
    )),

    PythonDemo(name='formula_recognition_demo', device_keys=['-d'],
               model_keys=['-m_encoder', '-m_decoder'], test_cases=combine_cases(
        TestCase(options={'--no_show': None}),
        [
            TestCase(options={
                '-i': str(OMZ_DIR / 'models/intel/formula-recognition-medium-scan-0001/'
                                    'assets/formula-recognition-medium-scan-0001.png'),
                '-m_encoder': ModelArg('formula-recognition-medium-scan-0001-im2latex-encoder'),
                '-m_decoder': ModelArg('formula-recognition-medium-scan-0001-im2latex-decoder'),
                '--vocab': ModelFileArg('formula-recognition-medium-scan-0001-im2latex-decoder', 'vocab.json'),
            }),
            TestCase(options={
                '-i': str(OMZ_DIR / 'models/intel/formula-recognition-polynomials-handwritten-0001/'
                                    'assets/formula-recognition-polynomials-handwritten-0001.png'),
                '-m_encoder': ModelArg('formula-recognition-polynomials-handwritten-0001-encoder'),
                '-m_decoder': ModelArg('formula-recognition-polynomials-handwritten-0001-decoder'),
                '--vocab': ModelFileArg('formula-recognition-polynomials-handwritten-0001-decoder', 'vocab.json'),
            })
        ],
    )),

    PythonDemo(name='gesture_recognition_demo', device_keys=['-d'],
               model_keys=['-m_d', '-m_a'], test_cases=combine_cases(
        TestCase(options={'--no_show': None,
                          '-i': TestDataArg('msasl/global_crops/_nz_sivss20/clip_0017/img_%05d.jpg'),
                          '-m_d': ModelArg('person-detection-asl-0001')}),
        [
            TestCase(options={'-m_a': ModelArg('asl-recognition-0004'), '-c': str(OMZ_DIR / 'data/dataset_classes/msasl100.json')}),
            TestCase(options={'-m_a': ModelArg('common-sign-language-0001'),
                              '-c': str(OMZ_DIR / 'data/dataset_classes/jester27.json')}),
            TestCase(options={'-m_a': ModelArg('common-sign-language-0002'),
                              '-c': str(OMZ_DIR / 'data/dataset_classes/common_sign_language12.json')}),
        ],
    )),

    PythonDemo(name='gpt2_text_prediction_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={
            '-i': ['The poem was written by'],
            '-m': ModelArg('gpt-2'),
            '-v': ModelFileArg('gpt-2', 'gpt2/vocab.json'),
            '--merges': ModelFileArg('gpt-2', 'gpt2/merges.txt'),
        }),
    )),

    PythonDemo(name='handwritten_text_recognition_demo', device_keys=['-d'], test_cases=combine_cases(
        [
            TestCase(options={
                '-i': str(OMZ_DIR / 'models/intel/handwritten-english-recognition-0001/assets/handwritten-english-recognition-0001.jpg'),
                '-m': ModelArg('handwritten-english-recognition-0001'),
                '-cl': str(OMZ_DIR / 'data/dataset_classes/gnhk.txt')
            }),
            TestCase(options={
                '-i': str(OMZ_DIR / 'models/intel/handwritten-japanese-recognition-0001/assets/handwritten-japanese-recognition-0001.png'),
                '-m': ModelArg('handwritten-japanese-recognition-0001'),
                '-cl': str(OMZ_DIR / 'data/dataset_classes/kondate_nakayosi.txt')
            }),
            TestCase(options={
                '-i': str(OMZ_DIR / 'models/intel/handwritten-simplified-chinese-recognition-0001/assets/handwritten-simplified-chinese-recognition-0001.png'),
                '-m': ModelArg('handwritten-simplified-chinese-recognition-0001'),
                '-cl': str(OMZ_DIR / 'data/dataset_classes/scut_ept.txt')
            }),
        ],
    )),

    PythonDemo(name='human_pose_estimation_3d_demo', device_keys=['-d'], test_cases=combine_cases(
       TestCase(options={'--no_show': None,
                         **MONITORS,
                         '-i': DataPatternArg('human-pose-estimation')}),
       TestCase(options={'-m': ModelArg('human-pose-estimation-3d-0001')}),
    )),

    PythonDemo(name='human_pose_estimation_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'-no_show': None,
            **MONITORS,
            '-i': DataPatternArg('human-pose-estimation')}),
        [
            TestCase({'-at': 'openpose', '-m': ModelArg('human-pose-estimation-0001')}),
            TestCase({'-at': 'higherhrnet', '-m': ModelArg('higher-hrnet-w32-human-pose-estimation')}),
            *combine_cases(
                TestCase(options={'-at': 'ae'}),
                single_option_cases('-m',
                    ModelArg('human-pose-estimation-0005'),
                    ModelArg('human-pose-estimation-0006'),
                    ModelArg('human-pose-estimation-0007'))),
        ],
    )),

    PythonDemo(name='image_inpainting_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'--no_show': None,
                          '-i': image_net_arg('00048311'),
                          '-m': ModelArg('gmcnn-places2-tf'),
                          '-ar': None})
    )),

    PythonDemo(name='image_retrieval_demo', device_keys=['-d'], test_cases=[
        # Test video can't be decoded by default Windows configuration, test -h instead
        TestCase({'-h': None}) if 'nt' == os.name else
        TestCase({
            '--no_show': None,
            **MONITORS,
            '-i': DATA_SEQUENCES['image-retrieval-video'],
            '-m': ModelArg('image-retrieval-0001'),
            '-g': image_retrieval_arg('gallery.txt')
        })
    ]),

    PythonDemo('image_translation_demo', ('--translation_model', '--segmentation_model'), test_cases=combine_cases(
        [
            TestCase({
                '--input_images': TestDataArg('coco128/images/train2017/'),
                '--reference_images': TestDataArg('coco128/images/train2017/')
            }),
            TestCase({
                '--input_images': TestDataArg('coco128/images/train2017/000000000009.jpg'),
                '--reference_images': TestDataArg('coco128/images/train2017/000000000025.jpg')
            }),
        ],
        TestCase({
            '--translation_model': ModelArg('cocosnet'),
            '--segmentation_model': ModelArg('hrnet-v2-c1-segmentation'),
            '-o': '.'
        }),
    )),

    PythonDemo(name='instance_segmentation_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'--no_show': None,
            **MONITORS,
            '-i': DataPatternArg('instance-segmentation'),
            '--labels': str(OMZ_DIR / 'data/dataset_classes/coco_80cl_bkgr.txt')}),
        single_option_cases('-m',
            ModelArg('instance-segmentation-security-0002'),
            ModelArg('instance-segmentation-security-0091'),
            ModelArg('instance-segmentation-security-0228'),
            ModelArg('instance-segmentation-security-1039'),
            ModelArg('instance-segmentation-security-1040')),
    )),

    PythonDemo(name='machine_translation_demo', device_keys=['-d'], test_cases=combine_cases(
        [
            TestCase(options={
                '-m': ModelArg('machine-translation-nar-de-en-0002'),
                '--tokenizer-src': ModelFileArg('machine-translation-nar-de-en-0002', 'tokenizer_src'),
                '--tokenizer-tgt': ModelFileArg('machine-translation-nar-de-en-0002', 'tokenizer_tgt'),
                '-i': [
                    'Der schnelle Braunfuchs springt über den faulen Hund.'
                    'Die fünf Boxzauberer springen schnell.',
                    'Dohlen lieben meine große Quarzsphinx.'
                ],
            }),
            TestCase(options={
                '-m': ModelArg('machine-translation-nar-en-de-0002'),
                '--tokenizer-src': ModelFileArg('machine-translation-nar-en-de-0002', 'tokenizer_src'),
                '--tokenizer-tgt': ModelFileArg('machine-translation-nar-en-de-0002', 'tokenizer_tgt'),
                '-i': [
                    'The quick brown fox jumps over the lazy dog.',
                    'The five boxing wizards jump quickly.',
                    'Jackdaws love my big sphinx of quartz.'
                ],
            }),
            TestCase(options={
                '-m': ModelArg('machine-translation-nar-en-ru-0002'),
                '--tokenizer-src': ModelFileArg('machine-translation-nar-en-ru-0002', 'tokenizer_src'),
                '--tokenizer-tgt': ModelFileArg('machine-translation-nar-en-ru-0002', 'tokenizer_tgt'),
                '-i': [
                    'The quick brown fox jumps over the lazy dog.',
                    'The five boxing wizards jump quickly.',
                    'Jackdaws love my big sphinx of quartz.'
                ],
            }),
            TestCase(options={
                '-m': ModelArg('machine-translation-nar-ru-en-0002'),
                '--tokenizer-src': ModelFileArg('machine-translation-nar-ru-en-0002', 'tokenizer_src'),
                '--tokenizer-tgt': ModelFileArg('machine-translation-nar-ru-en-0002', 'tokenizer_tgt'),
                '-i': [
                    'В чащах юга жил бы цитрус? Да, но фальшивый экземпляр!',
                    'Широкая электрификация южных губерний даст мощный толчок подъёму сельского хозяйства.',
                    'Съешь же ещё этих мягких французских булок да выпей чаю.'
                ],
            }),
        ]
    )),

    PythonDemo('monodepth_demo', test_cases=combine_cases(
        single_option_cases(
            '-m',
            ModelArg('fcrn-dp-nyu-depth-v2-tf'),
            ModelArg('midasnet'),
        ),
        TestCase({
            '-i': DataPatternArg('object-detection-demo'),
            **UTILIZATION_MONITORS_AND_NO_SHOW_COMMAND_LINE_OPTIONS
        })
    )),

    PythonDemo('multi_camera_multi_target_tracking_demo', model_keys=['-m', '--m_segmentation', '--m_reid'], test_cases=combine_cases(
        [
            TestCase({
                '--m_segmentation': ModelArg('instance-segmentation-security-0228'),
                '--m_reid': ModelArg('vehicle-reid-0001'),
                '--config': str(OMZ_DIR / 'demos/multi_camera_multi_target_tracking_demo/python/configs/vehicle.py'),
                '--output_video': 'multi_camera_multi_target_tracking_demo.avi',
            }),
            *combine_cases(
                single_option_cases(
                    '--m_reid',
                    ModelArg('person-reidentification-retail-0286'),
                    ModelArg('person-reidentification-retail-0287'),
                    ModelArg('person-reidentification-retail-0288'),
                ),
                TestCase({
                    '-m': ModelArg('person-detection-retail-0013'),
                    **MONITORS
                }),
            ),
        ],
        TestCase({
            '-i': [
                DataPatternArg('multi-camera-multi-target-tracking'),
                DataPatternArg('multi-camera-multi-target-tracking/repeated'),
            ],
            '--no_show': None,
        }),
    )),

    PythonDemo(name='noise_suppression_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'-i': TestDataArg('how_are_you_doing.wav')}),
        single_option_cases('-m',
            ModelArg('noise-suppression-denseunet-ll-0001'),
            ModelArg('noise-suppression-poconetlike-0001'))
    )),

    PythonDemo(name='object_detection_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'--no_show': None, **MONITORS, '-i': DataPatternArg('object-detection-demo')}),
        [
            *combine_cases(
                TestCase(options={'--architecture_type': 'centernet'}),
                [
                    *single_option_cases('-m',
                        ModelArg('ctdet_coco_dlav0_512'),
                    ),
                    *combine_cases(
                        TestCase(options={
                            '--mean_values': ['104.04', '113.985', '119.85'],
                            '--scale_values': ['73.695', '69.87', '70.89']
                        }),
                        single_option_cases('-m',
                            ModelFileArg('ctdet_coco_dlav0_512', 'ctdet_coco_dlav0_512.onnx'),
                        ),
                    ),
                ]
            ),
            *combine_cases(
                TestCase(options={'--architecture_type': 'faceboxes'}),
                [
                    TestCase(options={'-m': ModelArg('faceboxes-pytorch')}),
                    TestCase(options={'-m': ModelFileArg('faceboxes-pytorch', 'faceboxes-pytorch.onnx'),
                                      '--mean_values': ['104.0', '117.0', '123.0']}),
                ]
            ),
            TestCase(options={'--architecture_type': 'ctpn',
                              '-m': ModelArg('ctpn')}
            ),
            *combine_cases(
                TestCase(options={'--architecture_type': 'retinaface-pytorch'}),
                [
                    TestCase(options={'-m': ModelArg('retinaface-resnet50-pytorch')}),
                    TestCase(options={'-m': ModelFileArg('retinaface-resnet50-pytorch', 'retinaface-resnet50-pytorch.onnx'),
                                      '--mean_values': ['104.0', '117.0', '123.0']}),
                ]
            ),
            *combine_cases(
                TestCase(options={'--architecture_type': 'ssd'}),
                [
                    *single_option_cases('-m',
                        ModelArg('efficientdet-d0-tf'),
                        ModelArg('efficientdet-d1-tf'),
                        ModelArg('face-detection-0200'),
                        ModelArg('face-detection-0202'),
                        ModelArg('face-detection-0204'),
                        ModelArg('face-detection-0205'),
                        ModelArg('face-detection-0206'),
                        ModelArg('face-detection-adas-0001'),
                        ModelArg('face-detection-retail-0004'),
                        ModelArg('face-detection-retail-0005'),
                        ModelArg('faster_rcnn_inception_resnet_v2_atrous_coco'),
                        ModelArg('faster_rcnn_resnet50_coco'),
                        ModelArg('faster-rcnn-resnet101-coco-sparse-60-0001'),
                        ModelArg('pedestrian-and-vehicle-detector-adas-0001'),
                        ModelArg('pedestrian-detection-adas-0002'),
                        ModelArg('person-detection-0200'),
                        ModelArg('person-detection-0201'),
                        ModelArg('person-detection-0202'),
                        ModelArg('person-detection-retail-0013'),
                        ModelArg('person-vehicle-bike-detection-2000'),
                        ModelArg('person-vehicle-bike-detection-2001'),
                        ModelArg('person-vehicle-bike-detection-2002'),
                        ModelArg('person-vehicle-bike-detection-2003'),
                        ModelArg('person-vehicle-bike-detection-2004'),
                        ModelArg('product-detection-0001'),
                        ModelArg('rfcn-resnet101-coco-tf'),
                        ModelArg('retinanet-tf'),
                        ModelArg('ssd_mobilenet_v1_coco'),
                        ModelArg('ssd_mobilenet_v1_fpn_coco'),
                        ModelArg('ssd-resnet34-1200-onnx'),
                        ModelArg('ssdlite_mobilenet_v2'),
                        ModelArg('vehicle-detection-0200'),
                        ModelArg('vehicle-detection-0201'),
                        ModelArg('vehicle-detection-0202'),
                        ModelArg('vehicle-detection-adas-0002'),
                        ModelArg('vehicle-license-plate-detection-barrier-0106'),
                        ModelArg('person-detection-0106')),
                    TestCase(options={'-m': ModelFileArg('ssd-resnet34-1200-onnx', 'resnet34-ssd1200.onnx'),
                                      '--reverse_input_channels': None,
                                      '--mean_values': ['123.675', '116.28', '103.53'],
                                      '--scale_values': ['58.395', '57.12', '57.375']}),
                ]
            ),
            *combine_cases(
                TestCase(options={'--architecture_type': 'ultra_lightweight_face_detection'}),
                [
                    *single_option_cases('-m',
                        ModelArg('ultra-lightweight-face-detection-rfb-320'),
                        ModelArg('ultra-lightweight-face-detection-slim-320'),
                    ),
                    *combine_cases(
                        TestCase(options={
                            '--mean_values': ['127.0', '127.0', '127.0'],
                            '--scale_values': ['128.0', '128.0', '128.0']
                        }),
                        single_option_cases('-m',
                            ModelFileArg('ultra-lightweight-face-detection-rfb-320', 'ultra-lightweight-face-detection-rfb-320.onnx'),
                            ModelFileArg('ultra-lightweight-face-detection-slim-320', 'ultra-lightweight-face-detection-slim-320.onnx'),
                        ),
                    ),
                ]
            ),
            *combine_cases(
                TestCase(options={'--architecture_type': 'yolo'}),
                single_option_cases('-m',
                    ModelArg('mobilenet-yolo-v4-syg'),
                    ModelArg('person-vehicle-bike-detection-crossroad-yolov3-1020'),
                    ModelArg('yolo-v1-tiny-tf'),
                    ModelArg('yolo-v2-ava-0001'),
                    ModelArg('yolo-v2-ava-sparse-35-0001'),
                    ModelArg('yolo-v2-ava-sparse-70-0001'),
                    ModelArg('yolo-v2-tf'),
                    ModelArg('yolo-v2-tiny-ava-0001'),
                    ModelArg('yolo-v2-tiny-ava-sparse-30-0001'),
                    ModelArg('yolo-v2-tiny-ava-sparse-60-0001'),
                    ModelArg('yolo-v2-tiny-tf'),
                    ModelArg('yolo-v2-tiny-vehicle-detection-0001'),
                    ModelArg('yolo-v3-tf'),
                    ModelArg('yolo-v3-tiny-tf')),
            ),
            TestCase(options={'-at': 'yolov3-onnx', '-m': ModelArg('yolo-v3-onnx')}),
            TestCase(options={'-at': 'yolov3-onnx', '-m': ModelArg('yolo-v3-tiny-onnx')}),
            TestCase(options={'-at': 'yolov4', '-m': ModelArg('yolo-v4-tf')}),
            TestCase(options={'-at': 'yolov4', '-m': ModelArg('yolo-v4-tiny-tf')}),
            TestCase(options={'-at': 'yolof', '-m': ModelArg('yolof')}),
            *combine_cases(
                TestCase(options={'--architecture_type': 'detr'}),
                [
                    TestCase(options={'-m': ModelArg('detr-resnet50')}),
                    TestCase(options={'-m': ModelFileArg('detr-resnet50', 'detr-resnet50.onnx'),
                                     '--reverse_input_channels': None,
                                      '--mean_values': ['123.675', '116.28', '103.53'],
                                      '--scale_values': ['58.395', '57.12', '57.375']}),
                ]
            ),
            *combine_cases(
                TestCase(options={'--architecture_type': 'yolox'}),
                [
                    TestCase(options={'-m': ModelArg('yolox-tiny')}),
                    TestCase(options={'-m': ModelFileArg('yolox-tiny', 'yolox-tiny.onnx'),
                                      '--reverse_input_channels': None,
                                      '--mean_values': ['123.675', '116.28', '103.53'],
                                      '--scale_values': ['58.395', '57.12', '57.375']}),
                ]
            ),
            *combine_cases(
                TestCase(options={'--architecture_type': 'nanodet'}),
                [
                    TestCase(options={'-m': ModelArg('nanodet-m-1.5x-416')}),
                    TestCase(options={'-m': ModelFileArg('nanodet-m-1.5x-416', 'nanodet-m-1.5x-416.onnx'),
                                      '--mean_values': ['103.53', '116.28', '123.675'],
                                      '--scale_values': ['57.375', '57.12', '58.395']}),
                ]
            ),
            *combine_cases(
                TestCase(options={'--architecture_type': 'nanodet-plus'}),
                [
                    TestCase(options={'-m': ModelArg('nanodet-plus-m-1.5x-416')}),
                    TestCase(options={'-m': ModelFileArg('nanodet-plus-m-1.5x-416', 'nanodet-plus-m-1.5x-416.onnx'),
                                      '--mean_values': ['103.53', '116.28', '123.675'],
                                      '--scale_values': ['57.375', '57.12', '58.395']}),
                ]
            ),
        ],
    )),

    PythonDemo('place_recognition_demo', test_cases=combine_cases(
        single_option_cases('-m', ModelArg('netvlad-tf'), ModelFileArg('netvlad-tf', 'model_frozen.pb')),
        TestCase({
            '--input': TestDataArg('coco128/images/train2017/'),
            '--gallery_folder': TestDataArg('coco128/images/train2017/'),
            '--output': 'place_recognition_demo.avi',
            **UTILIZATION_MONITORS_AND_NO_SHOW_COMMAND_LINE_OPTIONS
        })
    )),

    PythonDemo(name='segmentation_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'--no_show': None, **MONITORS}),
        [
            TestCase(options={
                '-m': ModelArg('road-segmentation-adas-0001'),
                '-i': DataPatternArg('road-segmentation-adas'),
                '-at': 'segmentation',
            }),
            *combine_cases(
                TestCase(options={
                    '-i': DataPatternArg('semantic-segmentation-adas'),
                    '-at': 'segmentation',
                }),
                single_option_cases('-m',
                    ModelArg('semantic-segmentation-adas-0001'),
                    ModelArg('fastseg-large'),
                    ModelArg('fastseg-small'),
                    ModelArg('hrnet-v2-c1-segmentation'),
                    ModelArg('icnet-camvid-ava-0001'),
                    ModelArg('icnet-camvid-ava-sparse-30-0001'),
                    ModelArg('icnet-camvid-ava-sparse-60-0001'),
                    ModelArg('unet-camvid-onnx-0001'),
                    ModelArg('deeplabv3'),
                    ModelArg('pspnet-pytorch'),
                    ModelArg('drn-d-38'),
                    ModelArg('erfnet'),
                )),
            TestCase(options={
                '-m': ModelArg('f3net'),
                '-i': DataPatternArg('road-segmentation-adas'),
                '-at': 'salient_object_detection',
            }),
        ],
    )),

    PythonDemo(name='single_human_pose_estimation_demo', device_keys=['-d'],
              model_keys=['-m_od', '-m_hpe'], test_cases=combine_cases(
       TestCase(options={'--no_show': None, **MONITORS,
                          '-i': DataPatternArg('human-pose-estimation'),
                          '--person_label': '1'}),
       [
           *combine_cases(
               TestCase(options={'-m_hpe': ModelArg('single-human-pose-estimation-0001')}),
               single_option_cases('-m_od',
                   ModelArg('person-detection-retail-0013'),
                   ModelArg('ssd_mobilenet_v1_coco'))),
       ]
    )),

    PythonDemo(
        'smartlab_demo',
        model_keys=('--m_topall', '--m_topmove', '--m_sideall', '--m_sidemove', '--m_encoder', '--m_encoder_top', '--m_encoder_side', '--m_decoder'),
        test_cases=combine_cases(
            [
                TestCase({
                    '--mode': 'multiview',
                    '--m_encoder_top': ModelArg('smartlab-action-recognition-0001-encoder-top'),
                    '--m_encoder_side': ModelArg('smartlab-action-recognition-0001-encoder-side'),
                    '--m_decoder': ModelArg('smartlab-action-recognition-0001-decoder'),
                }),
                TestCase({
                    '--mode': 'mtcnn',
                    '--m_encoder': ModelArg('smartlab-sequence-modelling-0001'),
                    '--m_decoder': ModelArg('smartlab-sequence-modelling-0002')
                })
            ],
            TestCase({
                '--topview': TestDataArg('stream_1_top.mp4'),
                '--sideview': TestDataArg('stream_1_left.mp4'),
                '--m_topall': ModelArg('smartlab-object-detection-0001'),
                '--m_topmove': ModelArg('smartlab-object-detection-0002'),
                '--m_sideall': ModelArg('smartlab-object-detection-0003'),
                '--m_sidemove': ModelArg('smartlab-object-detection-0004'),
                '--no_show': None
            })
        )
    ),

    PythonDemo('sound_classification_demo', test_cases=combine_cases(
        single_option_cases(
            '-m',
            ModelArg('aclnet'),
            ModelArg('aclnet-int8'),
            ModelFileArg('aclnet-int8', 'aclnet_des_53_int8.onnx'),
        ),
        TestCase({'-i': TestDataArg('how_are_you_doing.wav')}),
    )),

    PythonDemo(name='speech_recognition_deepspeech_demo', device_keys=['-d'], test_cases=combine_cases(
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
                              '-L': ModelFileArg('mozilla-deepspeech-0.6.1', 'deepspeech-0.6.1-models/lm.binary')}),
            TestCase(options={'-p': 'mds08x_en',  # test online mode
                              '-m': ModelArg('mozilla-deepspeech-0.8.2'),
                              # run_tests.py puts pre-converted files into dl_dir as
                              # it always runs converter.py without --output_dir
                              '-L': ModelFileArg('mozilla-deepspeech-0.8.2', 'deepspeech-0.8.2-models.kenlm'),
                              '--realtime': None}),
            TestCase(options={'-p': 'mds08x_en',  # test without LM
                              '-m': ModelArg('mozilla-deepspeech-0.8.2')}),
        ],
    )),

    PythonDemo('speech_recognition_quartznet_demo', test_cases=combine_cases(
        single_option_cases(
            '-m',
            ModelArg('quartznet-15x5-en'),
            ModelFileArg('quartznet-15x5-en', 'quartznet.onnx'),
        ),
        TestCase({'-i': TestDataArg('how_are_you_doing.wav')}),
    )),

    PythonDemo(name='speech_recognition_wav2vec_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'-i': TestDataArg('how_are_you_doing.wav')}),
        single_option_cases('-m', ModelArg('wav2vec2-base'))
    )),

    PythonDemo(name='text_spotting_demo', device_keys=['-d'],
               model_keys=['-m_m', '-m_te', '-m_td'], test_cases=combine_cases(
        TestCase(options={'--no_show': None, '--delay': '1', **MONITORS,
                          '-i': DataPatternArg('text-detection')}),
        [
            TestCase(options={
                '-m_m': ModelArg('text-spotting-0005-detector'),
                '-m_te': ModelArg('text-spotting-0005-recognizer-encoder'),
                '-m_td': ModelArg('text-spotting-0005-recognizer-decoder'),
                '--no_track': None
            }),
        ]
    )),

    PythonDemo(name='text_to_speech_demo', device_keys=['-d'],
               model_keys=['-m_duration', '-m_forward', '-m_upsample', '-m_rnn', '-m_melgan'], test_cases=combine_cases(
        TestCase(options={'-i': [
                    'The quick brown fox jumps over the lazy dog.',
                    'The five boxing wizards jump quickly.'
                ]}),
        [
            TestCase(options={
                '-m_duration': ModelArg('forward-tacotron-duration-prediction'),
                '-m_forward': ModelArg('forward-tacotron-regression'),
                '-m_upsample': ModelArg('wavernn-upsampler'),
                '-m_rnn': ModelArg('wavernn-rnn')
            }),
            TestCase(options={
                '-m_duration': ModelArg('text-to-speech-en-0001-duration-prediction'),
                '-m_forward': ModelArg('text-to-speech-en-0001-regression'),
                '-m_melgan': ModelArg('text-to-speech-en-0001-generation')
            }),
            TestCase(options={
                '-m_duration': ModelArg('text-to-speech-en-multi-0001-duration-prediction'),
                '-m_forward': ModelArg('text-to-speech-en-multi-0001-regression'),
                '-m_melgan': ModelArg('text-to-speech-en-multi-0001-generation')
            }),
        ]
    )),

    PythonDemo(name='time_series_forecasting_demo', device_keys=[],
        model_keys=['-m'], test_cases=[TestCase(options={'-h': ''})]),

    PythonDemo(name='whiteboard_inpainting_demo', device_keys=['-d'],
               model_keys=['-m_i', '-m_s'], test_cases=combine_cases(
        TestCase(options={'-i': TestDataArg('msasl/global_crops/_nz_sivss20/clip_0017/img_%05d.jpg'),
                          **MONITORS,
                          '--no_show': None}),
        [
            *single_option_cases('-m_i',
                ModelArg('instance-segmentation-security-0002'),
                ModelArg('instance-segmentation-security-0228'),
                ModelArg('instance-segmentation-security-1039'),
                ModelArg('instance-segmentation-security-1040')),
            TestCase(options={'-m_s': ModelArg('semantic-segmentation-adas-0001')}),
        ]
    )),
]


BASE = { demo.subdirectory : deepcopy(demo) for demo in DEMOS }
