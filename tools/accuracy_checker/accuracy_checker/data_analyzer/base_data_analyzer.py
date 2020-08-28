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

import warnings
from ..dependency import ClassProvider
from ..logging import print_info


class BaseDataAnalyzer(ClassProvider):
    __provider_type__ = "data_analyzer"

    @classmethod
    def resolve(cls, name):
        if name not in cls.providers:
            warnings.warn("Suitable analyzer for {} not found. Default data analyzer is used".format(name))
            return BaseDataAnalyzer

        return cls.providers[name]

    @staticmethod
    def object_count(annotations):
        annotations_size = len(annotations)
        print_info('Total annotation objects: {size}'.format(size=annotations_size))
        return annotations_size


    def analyze(self, result: list, meta, count_objects=True):
        if count_objects:
            return {'annotations_size': self.object_count(result)}

        return {}
