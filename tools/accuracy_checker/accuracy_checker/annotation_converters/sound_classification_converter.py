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

from .format_converter import ConverterReturn, BaseFormatConverter
from ..config import PathField
from ..representation import ClassificationAnnotation
from ..utils import read_csv, get_path, check_file_existence


class SoundClassificationFormatConverter(BaseFormatConverter):
    __provider__ = 'sound_classification'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'annotation_file': PathField(description="Path to annotation in cvs format."),
            'audio_dir': PathField(
                is_directory=True, optional=True,
                description='Path to dataset audio files, used only for content existence check'
            )
        })

        return parameters

    def configure(self):
        self.annotation_file = self.get_value_from_config('annotation_file')
        self.audio_dir = self.get_value_from_config('audio_dir') or self.annotation_file.parent

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotation = []
        content_errors = [] if check_content else None
        original_annotation = read_csv(get_path(self.annotation_file), fieldnames=['identifier', 'label'])
        num_iterations = len(original_annotation)
        for audio_id, audio in enumerate(original_annotation):
            identifier = audio['identifier']
            label = int(audio['label'])
            if check_content:
                if not check_file_existence(self.audio_dir / identifier):
                    content_errors.append('{}: does not exist'.format(self.audio_dir / identifier))

            annotation.append(ClassificationAnnotation(identifier, label))
            if progress_callback is not None and audio_id % progress_interval == 0:
                progress_callback(audio_id / num_iterations * 100)

        return ConverterReturn(annotation, None, content_errors)
