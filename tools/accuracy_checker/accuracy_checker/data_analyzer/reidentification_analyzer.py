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
from .base_data_analyzer import BaseDataAnalyzer
from ..logging import print_info


class ReIdentificationClassificationDataAnalyzer(BaseDataAnalyzer):
    __provider__ = 'ReIdentificationClassificationAnnotation'

    def analyze(self, result: list, meta, count_objects=True):
        data_analysis = {}
        positive_pairs = 0
        negative_pairs = 0
        for data in result:
            positive_pairs += len(data.positive_pairs)
            negative_pairs += len(data.negative_pairs)
        all_pairs = positive_pairs + negative_pairs

        if count_objects:
            data_analysis['annotations_size'] = self.object_count(result)

        print_info('Total pairs: {}'.format(all_pairs))
        data_analysis['all_pairs'] = all_pairs
        print_info('Positive pairs: {}'.format(positive_pairs))
        data_analysis['positive_pairs'] = positive_pairs
        print_info('Negative pairs: {}'.format(negative_pairs))
        data_analysis['negative_pairs'] = negative_pairs

        return data_analysis


class ReIdentificationDataAnalyzer(BaseDataAnalyzer):
    __provider__ = 'ReIdentificationAnnotation'

    def analyze(self, result: list, meta, count_objects=True):
        data_analysis = {}
        person_in_query = Counter()
        person_in_gallery = Counter()
        camera_in_query = Counter()
        camera_in_gallery = Counter()
        unique_person = set()
        unique_camera = set()

        gallery_count = 0
        query_count = 0

        for data in result:
            if data.query:
                query_count += 1
                person_in_query.update([data.person_id])
                camera_in_query.update([data.camera_id])
            else:
                gallery_count += 1
                person_in_gallery.update([data.person_id])
                camera_in_gallery.update([data.camera_id])
            unique_person.add(data.person_id)
            unique_camera.add(data.camera_id)

        if count_objects:
            data_analysis['annotations_size'] = self.object_count(result)

        print_info('Number of elements in query: {}'.format(query_count))
        data_analysis['query_count'] = query_count

        print_info('Number of elements in gallery: {}'.format(gallery_count))
        data_analysis['gallery_count'] = gallery_count

        print_info('Number of unique objects: {}'.format(len(unique_person)))
        data_analysis['unique_objects'] = len(unique_person)
        data_analysis.update(self._collect_and_print_info_for_unique_elements(person_in_query, 'object', 'query'))
        data_analysis.update(self._collect_and_print_info_for_unique_elements(person_in_gallery, 'object', 'gallery'))

        print_info('Number of unique cameras: {}'.format(len(unique_camera)))
        data_analysis['unique_cameras'] = len(unique_camera)
        data_analysis.update(self._collect_and_print_info_for_unique_elements(camera_in_query, 'camera', 'query'))
        data_analysis.update(self._collect_and_print_info_for_unique_elements(camera_in_gallery, 'camera', 'gallery'))

        return data_analysis

    @staticmethod
    def _collect_and_print_info_for_unique_elements(counter, element_name, element_place):
        data_analysis = {}
        print_info('Count for each unique {name} in {place}:'.format(name=element_name, place=element_place))
        max_count = counter.most_common()[0][1]
        min_count = counter.most_common()[-1][1]
        most_common = []
        least_common = []
        for key, value in counter.most_common():
            print_info('{key}: {value}'.format(key=key, value=value))
            if value == max_count:
                most_common.append(key)
            if value == min_count:
                least_common.append(key)
        data_analysis['unique_{name}_{place}'.format(name=element_name,
                                                     place=element_place)] = dict(counter.most_common())
        print_info('Most common {name} in {place}: {value}'.format(name=element_name,
                                                                   place=element_place, value=most_common))
        data_analysis['most_common_{name}_in_{place}'.format(name=element_name, place=element_place)] = most_common
        print_info('Least common {name} in {place}: {value}'.format(name=element_name,
                                                                    place=element_place, value=least_common))
        data_analysis['least_common_{name}_in_{place}'.format(name=element_name, place=element_place)] = least_common

        return data_analysis
