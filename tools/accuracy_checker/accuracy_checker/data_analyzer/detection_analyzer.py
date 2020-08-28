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

from collections import Counter
import numpy as np
from .base_data_analyzer import BaseDataAnalyzer
from ..logging import print_info


class DetectionDataAnalyzer(BaseDataAnalyzer):
    __provider__ = 'DetectionAnnotation'

    def analyze(self, result: list, meta, count_objects=True):
        data_analysis = {}
        counter = Counter()
        all_boxes = 0
        diff_objects = 0
        width = 0
        height = 0
        width_diff = 0
        height_diff = 0
        size = len(result)
        for data in result:
            all_boxes += data.size
            width += np.sum(data.x_maxs - data.x_mins)
            height += np.sum(data.y_maxs - data.y_mins)
            if 'difficult_boxes' in data.metadata:
                diff_objects += len(data.metadata['difficult_boxes'])
                for i in data.metadata['difficult_boxes']:
                    width_diff += (data.x_maxs[i] - data.x_mins[i])
                    height_diff += (data.y_maxs[i] - data.y_mins[i])
            counter.update(data.labels)

        if count_objects:
            data_analysis['annotations_size'] = self.object_count(result)

        print_info('Total boxes {}'.format(all_boxes))
        data_analysis['all_boxes'] = all_boxes
        label_map = meta.get('label_map', {})

        for key in counter:
            if key in label_map:
                print_info('{name}: {value}'.format(name=label_map[key], value=counter[key]))
                data_analysis[label_map[key]] = counter[key]
            else:
                print_info('class_{key}: {value}'.format(key=key, value=counter[key]))
                data_analysis['class_{}'.format(key)] = counter[key]

        if size > 0:
            avg_num_diff_objects = diff_objects/size
            avg_num_all_objects = all_boxes/size
            avg_width = width/all_boxes
            avg_height = height/all_boxes

            print_info(
                'Average number of difficult objects (boxes) per image: {average}'.format(average=avg_num_diff_objects)
            )
            print_info(
                'Average number of objects (boxes) per image: {average}'.format(average=avg_num_all_objects)
            )
            print_info(
                'Average size detection object: width: {width}, '
                'height: {height}'.format(width=avg_width, height=avg_height)
            )

            data_analysis['average_num_difficult_objects'] = avg_num_diff_objects
            data_analysis['average_num_all_objects'] = avg_num_all_objects
            data_analysis['average_width'] = avg_width
            data_analysis['average_height'] = avg_height

            if diff_objects > 0:
                avg_width_diff_objects = width_diff/diff_objects
                avg_height_diff_objects = height_diff / diff_objects
                print_info(
                    'Average size difficult object: width: {width}, '
                    'height: {height}\n'.format(width=avg_width_diff_objects, height=avg_height_diff_objects)
                )
                data_analysis['average_width_difficult_objects'] = avg_width_diff_objects
                data_analysis['average_height_difficult_objects'] = avg_height_diff_objects

        return data_analysis
