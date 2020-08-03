"""
Copyright (c) 2018-2020 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import json
from accuracy_checker.serialize_parameters import fetch

def validate(json_dict):
    try:
        json.dumps(json_dict)
        return True
    except json.decoder.JSONDecodeError:
        return False

def check_topology_types(json_dict, topology_types):
    assert 'topology_type' in json_dict.keys()
    for topology_type in topology_types:
        assert topology_type in json_dict['topology_type']

def check_adapters(json_dict, adapters):
    assert 'datasets' in json_dict.keys()
    assert 'adapter' in json_dict['datasets'].keys()
    for adapter in adapters:
        assert adapter in json_dict['datasets']['adapter'].keys()

def check_converters(json_dict, converters):
    assert 'datasets' in json_dict.keys()
    assert 'converter' in json_dict['datasets'].keys()
    for converter in converters:
        assert converter in json_dict['datasets']['converter'].keys()

def check_launchers(json_dict, launchers):
    assert 'models' in json_dict.keys()
    assert 'launcher' in json_dict['models'].keys()
    for launcher in launchers:
        assert launcher in json_dict['models']['launcher'].keys()

class TestParameters:
    def test_all_parameters(self):
         json_dict = fetch()
         assert validate(json_dict) == True
         assert 'datasets' in json_dict.keys()
         dataset_content = json_dict['datasets'].keys()
         assert 'adapter' in dataset_content
         assert 'converter' in dataset_content
         assert 'metric' in dataset_content
         assert 'preprocessor' in dataset_content
         assert 'postprocessor' in dataset_content
         assert 'models' in json_dict.keys()
         models_content = json_dict['models'].keys()
         assert 'launcher' in models_content
         assert 'topology_type' in json_dict.keys()

    def test_image_classification(self):
         json_dict = fetch(topology_types=['image_classification'])
         assert validate(json_dict) == True
         check_topology_types(json_dict, ['image_classification'])
         check_adapters(json_dict, ['classification' ])
         check_converters(json_dict, [ 'imagenet' ])

    def test_object_detection(self):
         json_dict = fetch(topology_types=[ 'object_detection' ])
         assert validate(json_dict) == True
         check_topology_types(json_dict, [ 'object_detection' ])
         check_adapters(json_dict, [ 'ssd', 'tiny_yolo_v1', 'yolo_v2', 'yolo_v3' ])
         check_converters(json_dict, [ 'voc_detection' ])

    def test_yolo(self):
         json_dict = fetch(topology_types=[ 'yolo' ])
         assert validate(json_dict) == True
         check_topology_types(json_dict, [ 'yolo' ])
         check_adapters(json_dict, [ 'tiny_yolo_v1', 'yolo_v2', 'yolo_v3' ])
         check_converters(json_dict, [ 'voc_detection' ])

    def test_yolo_v1_tiny(self):
         json_dict = fetch(topology_types=[ 'yolo_v1_tiny' ])
         assert validate(json_dict) == True
         check_topology_types(json_dict, [ 'yolo_v1_tiny' ])
         check_adapters(json_dict, [ 'tiny_yolo_v1' ])
         check_converters(json_dict, [ 'voc_detection' ])

    def test_yolo_v2(self):
        json_dict = fetch(topology_types=[ 'yolo_v2' ])
        assert validate(json_dict) == True
        check_topology_types(json_dict, [ 'yolo_v2' ])
        check_adapters(json_dict, [ 'yolo_v2' ])
        check_converters(json_dict, [ 'voc_detection' ])

    def test_yolo_v2_tiny(self):
         json_dict = fetch(topology_types=[ 'yolo_v2_tiny' ])
         assert validate(json_dict) == True
         check_topology_types(json_dict, [ 'yolo_v2_tiny' ])
         check_adapters(json_dict, [ 'yolo_v2' ])
         check_converters(json_dict, [ 'voc_detection' ])

    def test_yolo_v3(self):
        json_dict = fetch(topology_types=[ 'yolo_v3' ])
        assert validate(json_dict) == True
        check_topology_types(json_dict, [ 'yolo_v3' ])
        check_adapters(json_dict, [ 'yolo_v3' ])
        check_converters(json_dict, [ 'voc_detection' ])

    def test_yolo_v3_tiny(self):
         json_dict = fetch(topology_types=[ 'yolo_v3_tiny' ])
         assert validate(json_dict) == True
         check_topology_types(json_dict, [ 'yolo_v3_tiny' ])
         check_adapters(json_dict, [ 'yolo_v3' ])
         check_converters(json_dict, [ 'voc_detection' ])

    def test_faster_rcnn(self):
         json_dict = fetch(topology_types=[ 'faster_rcnn' ])
         assert validate(json_dict) == True
         check_topology_types(json_dict, [ 'faster_rcnn' ])
         check_adapters(json_dict, [ 'ssd' ])
         check_converters(json_dict, [ 'voc_detection' ])

    def test_launchers(self):
         json_dict = fetch(launchers=[ 'dlsdk', 'opencv' ])
         assert validate(json_dict) == True
         check_launchers(json_dict, ['dlsdk', 'opencv'])
