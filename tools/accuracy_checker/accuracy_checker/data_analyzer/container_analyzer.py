"""
Copyright (c) 2018-2021 Intel Corporation

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


class ContainerDataAnalyzer(BaseDataAnalyzer):
    __provider__ = 'ContainerAnnotation'

    def analyze(self, result: list, meta, count_objects=True):
        data_analysis = {}
        if count_objects:
            data_analysis['annotations_size'] = self.object_count(result)
        dict_annotations = {}
        for container in result:
            for label_annotation in container.representations:
                if label_annotation in dict_annotations:
                    dict_annotations[label_annotation].append(container.representations[label_annotation])
                else:
                    dict_annotations[label_annotation] = [container.representations[label_annotation]]
        meta_names = {}
        if meta:
            for meta_name in meta:
                annotation_prefix = meta_name.split('_')[0]
                if annotation_prefix in meta_names:
                    meta_names[annotation_prefix].append(meta_name)
                else:
                    meta_names[annotation_prefix] = [meta_name]
        for label_annotation in dict_annotations:
            name_annotation = label_annotation.split('_')[0]
            first_element = next(iter(dict_annotations[label_annotation]), None)
            analyzer = BaseDataAnalyzer.provide(first_element.__class__.__name__)
            print_info('Analyzed annotation: {name}'.format(name=label_annotation))
            if name_annotation in meta_names:
                specific_keys = meta_names[name_annotation]
                annotation_specific_meta = {
                    key.split('{}_'.format(name_annotation))[-1]: meta[key] for key in specific_keys
                }
                data_analysis[label_annotation] = analyzer.analyze(dict_annotations[label_annotation],
                                                                   annotation_specific_meta, False)
            else:
                data_analysis[label_annotation] = analyzer.analyze(dict_annotations[label_annotation], meta, False)

        return data_analysis
