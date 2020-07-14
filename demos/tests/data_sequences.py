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

from args import image_net_arg, brats_arg, image_retrieval_arg

DATA_SEQUENCES = {
    'action-recognition': [
        image_net_arg('00000001'),
        image_net_arg('00000002'),
        image_net_arg('00000003'),
        image_net_arg('00000004'),
        image_net_arg('00000005'),
        image_net_arg('00000006'),
        image_net_arg('00000007'),
        image_net_arg('00000008'),
        image_net_arg('00000009'),
        image_net_arg('00000010'),
        image_net_arg('00000011'),
        image_net_arg('00000012'),
        image_net_arg('00000013'),
        image_net_arg('00000014'),
        image_net_arg('00000015'),
        image_net_arg('00000016'),
        image_net_arg('00000017'),
        image_net_arg('00000018'),
        image_net_arg('00000019'),
        image_net_arg('00000020'),
    ],

    'brain-tumor-nifti': [
        brats_arg('BRATS_485.nii.gz'),
    ],

    'face-detection-adas': [
        image_net_arg('00000002'),
        image_net_arg('00000032'),
        image_net_arg('00000184'),
        image_net_arg('00000442'),
        image_net_arg('00008165'),
        image_net_arg('00008170'),
        image_net_arg('00008172'),
        image_net_arg('00040548'),
        image_net_arg('00040557'),
        image_net_arg('00045630'),
    ],

    'face-recognition-gallery': [
        image_net_arg('00000184'),
        image_net_arg('00008165'),
        image_net_arg('00040548'),
    ],

    'gaze-estimation-adas': [
        image_net_arg('00008165'),
        image_net_arg('00008170'),
        image_net_arg('00012803'),
        image_net_arg('00018801'),
        image_net_arg('00020388'),
        image_net_arg('00021167'),
        image_net_arg('00033752'),
        image_net_arg('00040548'),
        image_net_arg('00040557'),
        image_net_arg('00044420'),
    ],

    'human-pose-estimation': [
        image_net_arg('00000002'),
        image_net_arg('00000184'),
        image_net_arg('00000442'),
        image_net_arg('00017291'),
        image_net_arg('00017293'),
        image_net_arg('00037129'),
        image_net_arg('00040548'),
        image_net_arg('00043066'),
        image_net_arg('00045630'),
        image_net_arg('00048311'),
    ],

    'classification': [
        image_net_arg('00000002'),
        image_net_arg('00000003'),
        image_net_arg('00000012'),
        image_net_arg('00000014'),
        image_net_arg('00000031'),
        image_net_arg('00000046'),
        image_net_arg('00000089'),
        image_net_arg('00000094'),
    ],

    'image-retrieval-video': [
        image_retrieval_arg('4946fb41-9da0-4af7-a858-b443bee6d0f6.dav'),
    ],

    'instance-segmentation': [
        image_net_arg('00000001'),
        image_net_arg('00000002'),
        image_net_arg('00000002'), # the demo has simple reid
        image_net_arg('00000003'),
        image_net_arg('00000004'),
        image_net_arg('00000008'),
        image_net_arg('00000010'),
        image_net_arg('00000017'),
        image_net_arg('00000019'),
        image_net_arg('00000020'),
    ],

    'multi-camera-multi-target-tracking': [
        image_net_arg('00000002'),
        image_net_arg('00000032'),
        image_net_arg('00017291'),
        image_net_arg('00017293'),
        image_net_arg('00040547'),
        image_net_arg('00000002'),
        image_net_arg('00000032'),
        image_net_arg('00017291'),
        image_net_arg('00017293'),
        image_net_arg('00040547'),
        image_net_arg('00000002'),
    ],

    'multi-camera-multi-target-tracking/repeated': [image_net_arg('00000002')] * 11,

    'object-detection-demo-ssd-async': [
        image_net_arg('00000001'),
        image_net_arg('00000002'),
        image_net_arg('00000003'),
        image_net_arg('00000004'),
        image_net_arg('00000005'),
        image_net_arg('00000006'),
        image_net_arg('00000007'),
        image_net_arg('00000008'),
        image_net_arg('00000014'),
        image_net_arg('00000018'),
        image_net_arg('00000022'),
        image_net_arg('00000023'),
        image_net_arg('00000032'),
    ],

    'person-detection-retail': [
        image_net_arg('00000002'),
        image_net_arg('00000002'),
        image_net_arg('00000002'),
        image_net_arg('00000002'),
        image_net_arg('00000002'),
        image_net_arg('00000032'),
        image_net_arg('00000002'),
        image_net_arg('00017291'),
        image_net_arg('00017293'),
        image_net_arg('00040547'),
    ],

    'person-vehicle-bike-detection-crossroad': [
        image_net_arg('00001012'),
        image_net_arg('00001892'),
        image_net_arg('00011595'),
        image_net_arg('00012792'),
        image_net_arg('00017291'),
        image_net_arg('00018042'),
        image_net_arg('00019585'),
        image_net_arg('00031320'),
        image_net_arg('00033757'),
        image_net_arg('00038629'),
    ],

    'road-segmentation-adas': [
        image_net_arg('00005809'),
        image_net_arg('00010401'),
        image_net_arg('00012792'),
        image_net_arg('00012796'),
        image_net_arg('00012799'),
        image_net_arg('00018799'),
        image_net_arg('00037112'),
        image_net_arg('00037128'),
        image_net_arg('00038629'),
        image_net_arg('00048316'),
    ],

    'semantic-segmentation-adas': [
        image_net_arg('00002790'),
        image_net_arg('00005809'),
        image_net_arg('00009780'),
        image_net_arg('00010401'),
        image_net_arg('00011595'),
        image_net_arg('00012792'),
        image_net_arg('00012799'),
        image_net_arg('00031329'),
        image_net_arg('00038629'),
        image_net_arg('00048316'),
    ],

    'single-image-super-resolution': [
        image_net_arg('00005409'),
    ],

    'smart-classroom-demo': [
        image_net_arg('00000074'),
        image_net_arg('00000164'),
        image_net_arg('00000181'),
        image_net_arg('00000164'),
        image_net_arg('00000181'),
        image_net_arg('00000001'),
        image_net_arg('00000074'),
    ],

    'text-detection': [
        image_net_arg('00000032'),
        image_net_arg('00001893'),
        image_net_arg('00008169'),
        image_net_arg('00010392'),
        image_net_arg('00010394'),
        image_net_arg('00010397'),
        image_net_arg('00010401'),
        image_net_arg('00028164'),
        image_net_arg('00028169'),
        image_net_arg('00030223'),
    ],

    'vehicle-license-plate-detection-barrier': [
        image_net_arg('00000014'),
        image_net_arg('00001892'),
        image_net_arg('00001910'),
        image_net_arg('00005809'),
        image_net_arg('00010401'),
        image_net_arg('00012792'),
        image_net_arg('00020382'),
        image_net_arg('00028190'),
        image_net_arg('00037128'),
        image_net_arg('00048316'),
    ],
}
