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

from ..utils import read_txt
from ..representation import ReIdentificationAnnotation
from ..config import PathField
from .format_converter import BaseFormatConverter, ConverterReturn


class ImageRetrievalConverter(BaseFormatConverter):
    __provider__ = 'image_retrieval'
    annotation_types = (ReIdentificationAnnotation, )

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'data_dir': PathField(description="dataset root directory", is_directory=True),
            'queries_annotation_file': PathField(
                description='txt-file with queries images and IDs concordance', optional=True
            ),
            'gallery_annotation_file': PathField(
                description='txt-file with gallery images and IDs concordance', optional=True
            )
        })

    def configure(self):
        self.data_dir = self.get_value_from_config('data_dir')
        self.queries_annotation_file = self.config.get(
            'queries_annotation_file', self.data_dir / 'queries' / 'list.txt'
        )
        self.gallery_annotation_file = self.config.get(
            'gallery_annotation_file', self.data_dir / 'gallery' / 'list.txt'
        )

    def convert(self,  *args, **kwargs):
        gallery = list()
        gallery_ids = set()
        for line in read_txt(self.gallery_annotation_file):
            identifier, id = line.split()
            gallery_ids.add(id)
            if '/' not in identifier:
                identifier = 'gallery/{}'.format(identifier)
            gallery.append(ReIdentificationAnnotation(identifier, 0, id, False))

        queries = list()
        queries_ids = set()
        for line in read_txt(self.queries_annotation_file):
            identifier, id = line.split()
            queries_ids.add(id)
            if '/' not in identifier:
                identifier = 'queries/{}'.format(identifier)
            queries.append(ReIdentificationAnnotation(identifier, 1, id, True))

        meta = {'num_identities': len(queries_ids | gallery_ids)}

        return ConverterReturn(gallery + queries, meta, None)
