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
        areas = {}
        for data in result:
            total_instances += data.size
            counter.update(data.labels)
            for label, rect in zip(data.labels, data.metadata['rects']):
                if areas.get(label):
                    areas[label]['area'].append(float(rect[2] * rect[3]))
                    areas[label]['width'].append(float(rect[2]))
                    areas[label]['height'].append(float(rect[3]))
                else:
                    areas[label]={'area': [float(rect[2] * rect[3])], 'width': [float(rect[2])], 'height': [float(rect[3])]}

        for key in areas:
            size = counter[key]
            areas[key]['area'] = sum(areas[key]['area']) / size
            areas[key]['width'] = sum(areas[key]['width']) / size
            areas[key]['height'] = sum(areas[key]['height']) / size

        print_info('Total instances: {value}'.format(value=total_instances))
        data_analysis['total_instances'] = total_instances
        label_map = meta.get('label_map', {})

        for key in counter:
            if key in label_map:
                print_info('{name}: count = {count}, average area = {area}, '
                           'average width = {width}, average height = {height}'.format(name=label_map[key],
                                                                                       count=counter[key],
                                                                                       area=areas[key]['area'],
                                                                                       width=areas[key]['width'],
                                                                                       height=areas[key]['height'],))
                data_analysis[label_map[key]] = {'count': counter[key], 'average_area': areas[key]['area'],
                           'average_width': areas[key]['width'], 'average_height': areas[key]['height']}
            else:
                print_info('class_{key}: count = {count}, average area = {area}, '
                           'average width = {width}, average height = {height}'.format(key=key,
                                                                                       count=counter[key],
                                                                                       area=areas[key]['area'],
                                                                                       width=areas[key]['width'],
                                                                                       height=areas[key]['height'],))
                data_analysis['class_{key}'.format(key=key)] = {'count': counter[key], 'average_area': areas[key]['area'],
                           'average_width': areas[key]['width'], 'average_height': areas[key]['height']}

        return data_analysis
