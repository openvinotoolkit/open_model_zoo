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
            ModelArg('person-reidentification-retail-0277'),
            ModelArg('person-reidentification-retail-0286'),
            ModelArg('person-reidentification-retail-0287'),
            ModelArg('person-reidentification-retail-0288')),
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
            ModelArg('resnet-50-caffe2')),
    )),

    NativeDemo(subdirectory='interactive_face_detection_demo',
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

    NativeDemo(subdirectory='object_detection_demo', device_keys=[], test_cases=combine_cases(
        TestCase(options={'--no_show': None,
            **MONITORS,
            '-i': DataPatternArg('object-detection-demo')}),
        [
            *combine_cases(
                TestCase(options={'-at': 'ssd'}),
                single_option_cases('-m',
                    ModelArg('face-detection-adas-0001'),
                    ModelArg('face-detection-retail-0004'),
                    ModelArg('face-detection-retail-0005'),
                    ModelArg('face-detection-retail-0044'),
                    ModelArg('pedestrian-and-vehicle-detector-adas-0001'),
                    ModelArg('pedestrian-detection-adas-0002'),
                    ModelArg('pelee-coco'),
                    ModelArg('person-detection-0200'),
                    ModelArg('person-detection-0201'),
                    ModelArg('person-detection-0202'),
                    ModelArg('person-detection-retail-0013'),
                    ModelArg('vehicle-detection-adas-0002'),
                    ModelArg('vehicle-license-plate-detection-barrier-0106'),
                    ModelArg('vehicle-license-plate-detection-barrier-0123'))),
            *combine_cases(
                TestCase(options={'-at': 'yolo'}),
                single_option_cases('-m',
                    ModelArg('yolo-v3-tf'),
                    ModelArg('yolo-v3-tiny-tf'))),
        ],
        )
    ),

    NativeDemo('pedestrian_tracker_demo', device_keys=['-d_det', '-d_reid'], test_cases=combine_cases(
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

    NativeDemo(subdirectory='segmentation_demo_async', device_keys=['-d'], test_cases=combine_cases(
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

    PythonDemo(subdirectory='bert_question_answering_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'-i': 'https://en.wikipedia.org/wiki/OpenVINO',
                          '--questions': ['What frameworks does OpenVINO support?', 'Who are developers?']}),
        [
            TestCase(options={
                '-m': ModelArg('bert-small-uncased-whole-word-masking-squad-0001'),
                '--input_names': 'input_ids,attention_mask,token_type_ids',
                '--output_names': 'output_s,output_e',
                '--vocab': str(OMZ_DIR / 'models/intel/bert-small-uncased-whole-word-masking-squad-0001/vocab.txt'),
            }),
            TestCase(options={
                '-m': ModelArg('bert-small-uncased-whole-word-masking-squad-0002'),
                '--input_names': 'input_ids,attention_mask,token_type_ids,position_ids',
                '--output_names': 'output_s,output_e',
                '--vocab': str(OMZ_DIR / 'models/intel/bert-small-uncased-whole-word-masking-squad-0002/vocab.txt'),
            }),
        ]
    )),

    PythonDemo(subdirectory='bert_question_answering_embedding_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'-i': 'https://en.wikipedia.org/wiki/OpenVINO',
                          '--questions': ['What frameworks does OpenVINO support?', 'Who are developers?']}),
        [
            TestCase(options={
                '-m_emb': ModelArg('bert-large-uncased-whole-word-masking-squad-emb-0001'),
                '--input_names_emb': 'input_ids,attention_mask,token_type_ids,position_ids',
                '--vocab': str(OMZ_DIR / 'models/intel/bert-large-uncased-whole-word-masking-squad-emb-0001/vocab.txt'),
                '-m_qa': ModelArg('bert-small-uncased-whole-word-masking-squad-0001'),
                '--input_names_qa': 'input_ids,attention_mask,token_type_ids',
                '--output_names_qa': 'output_s,output_e',
            }),
            TestCase(options={
                '-m_emb': ModelArg('bert-large-uncased-whole-word-masking-squad-emb-0001'),
                '--input_names_emb': 'input_ids,attention_mask,token_type_ids,position_ids',
                '--vocab': str(OMZ_DIR / 'models/intel/bert-large-uncased-whole-word-masking-squad-emb-0001/vocab.txt'),
            }),
        ]
    )),

    PythonDemo(subdirectory='colorization_demo', device_keys=['-d'], test_cases=combine_cases(
       TestCase(options={
           '--no_show': None,
           **MONITORS,
           '-i': DataPatternArg('classification'),
           '-m': ModelArg('colorization-v2'),
       })
    )),

    PythonDemo(subdirectory='gesture_recognition_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'--no_show': None,
                          '-i': TestDataArg('msasl/global_crops/_nz_sivss20/clip_0017/img_%05d.jpg'),
                          '-m_d': ModelArg('person-detection-asl-0001')}),
        [
            TestCase(options={'-m_a': ModelArg('asl-recognition-0004'), '-c': DemoFileArg('msasl100-classes.json')}),
            TestCase(options={'-m_a': ModelArg('common-sign-language-0001'),
                              '-c': DemoFileArg('jester27-classes.json')}),
        ],
    )),

    PythonDemo(subdirectory='human_pose_estimation_3d_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'--no_show': None,
                          **MONITORS,
                          '-i': DataPatternArg('human-pose-estimation')}),
        TestCase(options={'-m': ModelArg('human-pose-estimation-3d-0001')}),
    )),

    PythonDemo(subdirectory='image_inpainting_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'--no_show': None,
                          '-i': image_net_arg('00048311'),
                          '-m': ModelArg('gmcnn-places2-tf'),
                          '-ar': None})
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

    PythonDemo(subdirectory='machine_translation_demo', device_keys=[], test_cases=combine_cases(
       [
           TestCase(options={
               '-m': ModelArg('machine-translation-nar-en-ru-0001'),
               '--tokenizer-src': str(OMZ_DIR / 'models/intel/machine-translation-nar-en-ru-0001/tokenizer_src'),
               '--tokenizer-tgt': str(OMZ_DIR / 'models/intel/machine-translation-nar-en-ru-0001/tokenizer_tgt'),
               '--output-name': 'pred',
               '-i': [
                   'The quick brown fox jumps over the lazy dog.',
                   'The five boxing wizards jump quickly.',
                   'Jackdaws love my big sphinx of quartz.'
               ],
           }),
           TestCase(options={
               '-m': ModelArg('machine-translation-nar-ru-en-0001'),
               '--tokenizer-src': str(OMZ_DIR / 'models/intel/machine-translation-nar-ru-en-0001/tokenizer_src'),
               '--tokenizer-tgt': str(OMZ_DIR / 'models/intel/machine-translation-nar-ru-en-0001/tokenizer_tgt'),
               '--output-name': 'pred',
               '-i': [
                   'В чащах юга жил бы цитрус? Да, но фальшивый экземпляр!',
                   'Широкая электрификация южных губерний даст мощный толчок подъёму сельского хозяйства.',
                   'Съешь же ещё этих мягких французских булок да выпей чаю.'
               ],
           }),
       ]
    )),

    PythonDemo(subdirectory='monodepth_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'-i': image_net_arg('00000002'),
                          '-m': ModelArg('midasnet')})
    )),

    PythonDemo(subdirectory='multi_camera_multi_target_tracking', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'--no_show': None,
            **MONITORS,
            '-i': [DataPatternArg('multi-camera-multi-target-tracking'),
                DataPatternArg('multi-camera-multi-target-tracking/repeated')],
            '-m': ModelArg('person-detection-retail-0013')}),
        single_option_cases('--m_reid',
            ModelArg('person-reidentification-retail-0277'),
            ModelArg('person-reidentification-retail-0286'),
            ModelArg('person-reidentification-retail-0287'),
            ModelArg('person-reidentification-retail-0288')),
    )),

    PythonDemo(subdirectory='object_detection_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'--no_show': None, **MONITORS, '-i': DataPatternArg('object-detection-demo')}),
        [
            *combine_cases(
                TestCase(options={'--architecture_type': 'ssd'}),
                single_option_cases('-m',
                    ModelArg('face-detection-0200'),
                    ModelArg('face-detection-0202'),
                    ModelArg('face-detection-0204'),
                    ModelArg('face-detection-0205'),
                    ModelArg('face-detection-0206'),
                    ModelArg('face-detection-adas-0001'),
                    ModelArg('face-detection-retail-0004'),
                    ModelArg('face-detection-retail-0005'),
                    ModelArg('face-detection-retail-0044'),
                    ModelArg('pedestrian-and-vehicle-detector-adas-0001'),
                    ModelArg('pedestrian-detection-adas-0002'),
                    ModelArg('person-detection-0106'),
                    ModelArg('person-detection-0200'),
                    ModelArg('person-detection-0201'),
                    ModelArg('person-detection-0202'),
                    ModelArg('person-detection-retail-0013'),
                    ModelArg('pelee-coco'),
                    ModelArg('product-detection-0001'),
                    ModelArg('retinanet-tf'),
                    ModelArg('ssd-resnet34-1200-onnx'),
                    ModelArg('vehicle-detection-adas-0002'),
                    ModelArg('vehicle-license-plate-detection-barrier-0106')),
            ),
            *combine_cases(
                TestCase(options={'--architecture_type': 'yolo'}),
                single_option_cases('-m',
                    ModelArg('mobilefacedet-v1-mxnet'),
                    ModelArg('yolo-v1-tiny-tf'),
                    ModelArg('yolo-v2-tiny-tf'),
                    ModelArg('yolo-v2-tiny-vehicle-detection-0001'),
                    ModelArg('yolo-v2-tf'),
                    ModelArg('yolo-v3-tf')),
            ),
            *combine_cases(
                TestCase(options={'--architecture_type': 'centernet'}),
                single_option_cases('-m',
                    ModelArg('ctdet_coco_dlav0_384'),
                    ModelArg('ctdet_coco_dlav0_512')),
            ),
            *combine_cases(
                TestCase(options={'--architecture_type': 'faceboxes',
                                  '-m': ModelArg('faceboxes-pytorch')})
            ),
            *combine_cases(
                TestCase(options={'--architecture_type': 'retina'}),
                single_option_cases('-m',
                    ModelArg('retinaface-anti-cov'),
                    ModelArg('retinaface-resnet50'))
            ),
        ],
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

    PythonDemo(subdirectory='single_human_pose_estimation_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'--no_show': None, **MONITORS,
                           '-i': DataPatternArg('human-pose-estimation'),
                           '--person_label': '1'}),
        [
            *combine_cases(
                TestCase(options={'-m_hpe': ModelArg('single-human-pose-estimation-0001')}),
                single_option_cases('-m_od',
                    ModelArg('mobilenet-ssd'),
                    ModelArg('person-detection-retail-0013'),
                    ModelArg('ssd_mobilenet_v1_coco'))),
        ]
    )),

    PythonDemo(subdirectory='text_spotting_demo', device_keys=['-d'], test_cases=combine_cases(
        TestCase(options={'--no_show': None, '--delay': '1', **MONITORS,
                          '-i': DataPatternArg('text-detection')}),
        [
            TestCase(options={
                '-m_m': ModelArg('text-spotting-0003-detector'),
                '-m_te': ModelArg('text-spotting-0003-recognizer-encoder'),
                '-m_td': ModelArg('text-spotting-0003-recognizer-decoder'),
                '--no_track': None
            }),
        ]
    )),
]

DEMOS = NATIVE_DEMOS + PYTHON_DEMOS
