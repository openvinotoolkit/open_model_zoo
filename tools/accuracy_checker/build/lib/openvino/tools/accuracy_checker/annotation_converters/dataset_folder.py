"""
Copyright (c) 2018-2022 Intel Corporation

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

from .format_converter import DirectoryBasedAnnotationConverter, ConverterReturn
from ..representation import ClassificationAnnotation


class DatasetFolderConverter(DirectoryBasedAnnotationConverter):
    __provider__ = 'cls_dataset_folder'

    def convert(self, check_content=False, **kwargs):
        meta = self.get_meta()
        annotations = []
        for idx, cls_dir in meta['label_map'].items():
            for img in (self.data_dir / cls_dir).glob('*'):
                identifier = '{}/{}'.format(cls_dir, img.name)
                annotations.append(ClassificationAnnotation(identifier, idx))

        return ConverterReturn(annotations, meta, None)

    def get_meta(self):
        classes = [directory.name for directory in self.data_dir.glob('*') if directory.is_dir()]
        classes.sort()
        return {'label_map': dict(enumerate(classes))}
