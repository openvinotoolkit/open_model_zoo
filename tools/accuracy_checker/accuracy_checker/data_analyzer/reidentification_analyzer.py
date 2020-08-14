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

from collections import defaultdict, Counter
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

        print_info('Number of unique objects: {}'.format(len(unique_person)))
        data_analysis['unique_objects'] = len(unique_person)
        print_info('Count for each unique object in query:')
        for key in person_in_query.keys():
            print_info('{key}: {count}'.format(key=key, count=person_in_query[key]))
        data_analysis['unique_objects_query'] = dict(person_in_query)
        print_info('Count for each unique object in gallery:')
        for key in person_in_gallery.keys():
            print_info('{key}: {count}'.format(key=key, count=person_in_gallery[key]))
        data_analysis['unique_objects_gallery'] = dict(person_in_gallery)

        print_info('Number of unique cameras: {}'.format(len(unique_camera)))
        data_analysis['unique_cameras'] = len(unique_camera)
        print_info('Count for each unique camera in query:')
        for key in camera_in_query.keys():
            print_info('{key}: {count}'.format(key=key, count=camera_in_query[key]))
        data_analysis['unique_cameras_query'] = dict(camera_in_query)
        print_info('Count for each unique camera in gallery:')
        for key in camera_in_gallery.keys():
            print_info('{key}: {count}'.format(key=key, count=camera_in_gallery[key]))
        data_analysis['unique_cameras_gallery'] = dict(camera_in_gallery)

        print_info('Number of elements in query: {}'.format(query_count))
        data_analysis['query_count'] = query_count
        print_info('Number of elements in gallery: {}'.format(gallery_count))
        data_analysis['gallery_count'] = gallery_count

        return data_analysis
