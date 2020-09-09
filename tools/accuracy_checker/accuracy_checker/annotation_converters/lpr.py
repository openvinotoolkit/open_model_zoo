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

from ..config import PathField
from ..representation import CharacterRecognitionAnnotation
from ..utils import read_txt, check_file_existence
from .format_converter import BaseFormatConverter, ConverterReturn


class LPRConverter(BaseFormatConverter):
    __provider__ = 'lpr_txt'
    annotation_types = (CharacterRecognitionAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'annotation_file': PathField(description="Path to annotation (.txt)."),
            'decoding_dictionary_file': PathField(
                optional=True, description="Path to file containing dictionary for output decoding."
            ),
            'images_dir': PathField(
                is_directory=True, optional=True,
                description='path to dataset images, used only for content existence check'
            )
        })

        return configuration_parameters

    def configure(self, *args, **kwargs):
        self.annotation_file = self.get_value_from_config('annotation_file')
        self.meta = prepare_meta(self.get_value_from_config('decoding_dictionary_file'))
        self.images_dir = self.get_value_from_config('images_dir') or self.annotation_file.parent

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotations = []
        read_annotation = read_txt(self.annotation_file)
        num_iterations = len(read_annotation)
        content_errors = [] if check_content else None
        for line_id, line in enumerate(read_annotation):
            line_rep = line.split(' ')
            identifier = line_rep[0]
            if check_content:
                if not check_file_existence(self.images_dir / identifier):
                    content_errors.append('{}: does not exist'.format(self.images_dir / identifier))
            if progress_callback and line_id % progress_interval == 0:
                progress_callback(line_id * 100 / num_iterations)

            label = line_rep[1]
            if label == 'NA':
                continue
            annotations.append(CharacterRecognitionAnnotation(identifier, label))

        return ConverterReturn(annotations, self.meta, content_errors)


def prepare_meta(meta_dict):
    label_map = {}
    for line in read_txt(meta_dict):
        key_val = line.split(' ')
        label_map[int(key_val[0])] = key_val[1]

    return {'label_map': label_map}
