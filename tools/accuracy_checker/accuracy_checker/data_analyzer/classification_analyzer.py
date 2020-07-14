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

from collections import Counter
from .base_data_analyzer import BaseDataAnalyzer
from ..logging import print_info


class ClassificationDataAnalyzer(BaseDataAnalyzer):
    __provider__ = 'ClassificationAnnotation'

    def analyze(self, result: list, meta, count_objects=True):
        counter = Counter()
        class_dict = {}
        label_map = (meta or {}).get('label_map', {})
        data_analysis = {}
        if count_objects:
            data_analysis['annotations_size'] = self.object_count(result)
        print_info('Total objects per class:')
        for data in result:
            counter.update([data.label])
        for key in counter:
            if key in label_map:
                class_dict[key] = label_map[key]
            else:
                class_dict[key] = 'class {key}'.format(key=key)
            print_info('{class_name}: {value}'.format(class_name=class_dict[key], value=counter[key]))
            data_analysis[class_dict[key]] = counter[key]

        return data_analysis
