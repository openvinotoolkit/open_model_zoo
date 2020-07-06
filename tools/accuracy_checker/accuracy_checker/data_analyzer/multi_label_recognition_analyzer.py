"""
Copyright (c) 2019 Intel Corporation

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

import numpy as np
from .base_data_analyzer import BaseDataAnalyzer
from ..logging import print_info


class MultiLabelRecognitionDataAnalyzer(BaseDataAnalyzer):
    __provider__ = 'MultiLabelRecognitionAnnotation'

    def analyze(self, result: list, meta, count_objects=True):
        data_analysis = {}
        if count_objects:
            data_analysis['annotations_size'] = self.object_count(result)
        count = np.zeros_like(result[0].multi_label)
        ignored_objects = np.zeros_like(result[0].multi_label)
        label_map = None
        if meta:
            label_map = meta.get('label_map', {})
        if not label_map:
            label_map = {i: 'class {}'.format(i) for i in range(result[0].multi_label.size)}
        for data in result:
            count += data.multi_label > 0
            ignored_objects += data.multi_label == -1
        for key in label_map:
            print_info('{name}: {value}'.format(name=label_map[key], value=count[key]))
            data_analysis[label_map[key]] = int(count[key])
        ignored_instances = []
        for key in label_map:
            print_info('ignored instances {name}: {value}'.format(name=label_map[key], value=ignored_objects[key]))
            ignored_instances.append(int(ignored_objects[key]))
        data_analysis['ignored_instances'] = ignored_instances

        return data_analysis
