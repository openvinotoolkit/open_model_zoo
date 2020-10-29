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

from ..utils import read_txt, check_file_existence
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
        return params

    def configure(self):
        self.data_dir = self.get_value_from_config('data_dir')
        self.queries_annotation_file = self.config.get(
            'queries_annotation_file', self.data_dir / 'queries' / 'list.txt'
        )
        self.gallery_annotation_file = self.config.get(
            'gallery_annotation_file', self.data_dir / 'gallery' / 'list.txt'
        )

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        content_errors = None if not check_content else []
        gallery = []
        gallery_ids = set()
        if progress_callback:
            num_iteration = len(read_txt(self.gallery_annotation_file)) + len(read_txt(self.queries_annotation_file))
        for line_id, line in enumerate(read_txt(self.gallery_annotation_file)):
            identifier, image_id = line.split()
            gallery_ids.add(image_id)
            if '/' not in identifier:
                identifier = 'gallery/{}'.format(identifier)
            if check_content:
                if not check_file_existence(self.data_dir / identifier):
                    content_errors.append('{}: does not exist'.format(self.data_dir / identifier))

            gallery.append(ReIdentificationAnnotation(identifier, 0, image_id, False))

            if progress_callback and line_id % progress_interval == 0:
                progress_callback(line_id * 100 / num_iteration)

        queries = []
        queries_ids = set()
        for line_id, line in enumerate(read_txt(self.queries_annotation_file)):
            identifier, image_id = line.split()
            queries_ids.add(image_id)
            if '/' not in identifier:
                identifier = 'queries/{}'.format(identifier)
            if check_content:
                if not check_file_existence(self.data_dir / identifier):
                    content_errors.append('{}: does not exist'.format(self.data_dir / identifier))

            if progress_callback and line_id + len(gallery) % progress_interval == 0:
                progress_callback((line_id + len(gallery)) * 100 / num_iteration)

            queries.append(ReIdentificationAnnotation(identifier, 1, image_id, True))

        meta = {'num_identities': len(queries_ids | gallery_ids)}

        return ConverterReturn(gallery + queries, meta, content_errors)
