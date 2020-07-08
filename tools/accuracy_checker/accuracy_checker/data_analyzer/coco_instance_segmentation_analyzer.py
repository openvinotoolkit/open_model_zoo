"""
Copyright (c) 2020 Intel Corporation

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


class CoCoInstanceSegmentationDataAnalyzer(BaseDataAnalyzer):
    __provider__ = 'CoCoInstanceSegmentationAnnotation'

    def analyze(self, result: list, meta, count_objects=True):
        data_analysis = {}
        if count_objects:
            data_analysis['annotations_size'] = self.object_count(result)

        counter = Counter()
        total_instances = 0
        characteristics = {}
        for data in result:
            total_instances += data.size
            counter.update(data.labels)
            for label, rect, area in zip(data.labels, data.metadata['rects'], data.areas):
                if characteristics.get(label):
                    characteristics[label]['area'].append(float(area))
                    characteristics[label]['width'].append(float(rect[2]))
                    characteristics[label]['height'].append(float(rect[3]))
                else:
                    characteristics[label] = {'area': [float(area)],
                                              'width': [float(rect[2])], 'height': [float(rect[3])]}

        for key in characteristics:
            size = counter[key]
            characteristics[key]['area'] = {'average': sum(characteristics[key]['area']) / size,
                                            'min': min(characteristics[key]['area']),
                                            'max': max(characteristics[key]['area'])}
            characteristics[key]['width'] = {'average': sum(characteristics[key]['width']) / size,
                                             'min': min(characteristics[key]['width']),
                                             'max': max(characteristics[key]['width'])}
            characteristics[key]['height'] = {'average': sum(characteristics[key]['height']) / size,
                                              'min': min(characteristics[key]['height']),
                                              'max': max(characteristics[key]['height'])}

        print_info('Total instances: {value}'.format(value=total_instances))
        data_analysis['total_instances'] = total_instances
        label_map = meta.get('label_map', {})

        for key in counter:
            class_name = label_map.get(key, 'class_{key}'.format(key=key))
            print_info('{class_name}: count = {count}, area = {area}, width = {width}, height = {height}'.format(
                class_name=class_name,
                count=counter[key],
                area=characteristics[key]['area'],
                width=characteristics[key]['width'],
                height=characteristics[key]['height'],))
            data_analysis[class_name] = {'count': counter[key], 'area': characteristics[key]['area'],
                                         'width': characteristics[key]['width'],
                                         'height': characteristics[key]['height']}

        return data_analysis
