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
