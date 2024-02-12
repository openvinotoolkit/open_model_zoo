"""
Copyright (c) 2018-2024 Intel Corporation

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

from .format_converter import FileBasedAnnotationConverter, ConverterReturn
from ..utils import read_txt, OrderedSet
from ..representation import ReIdentificationClassificationAnnotation
from ..config import NumberField


class SpeakerReIdentificationDatasetConverter(FileBasedAnnotationConverter):
    __provider__ = 'speaker_reidentification'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({'max_pairs': NumberField(optional=True, value_type=int)})
        return params

    def configure(self):
        super().configure()
        self.max_pairs = self.get_value_from_config('max_pairs')

    def convert(self, check_content=False, **kwargs):
        annotations = []
        positive_pairs, negative_pairs = {}, {}
        audio_files = OrderedSet()
        for line in read_txt(self.annotation_file):
            is_positive, audio1, audio2 = line.split()
            if int(is_positive):
                if audio1 not in positive_pairs:
                    positive_pairs[audio1] = []
                positive_pairs[audio1].append(audio2)
            else:
                if audio1 not in negative_pairs:
                    negative_pairs[audio1] = []
                negative_pairs[audio1].append(audio2)
            audio_files.add(audio1)
            audio_files.add(audio2)
        for audio in audio_files:
            audio_positive = positive_pairs.get(audio, [])
            audio_negative = negative_pairs.get(audio, [])
            if self.max_pairs:
                if len(audio_positive) > self.max_pairs:
                    audio_positive = audio_positive[:self.max_pairs]
                if len(audio_negative) > self.max_pairs:
                    audio_negative = audio_negative[:self.max_pairs]
            annotations.append(ReIdentificationClassificationAnnotation(
                audio, audio_positive, audio_negative))
        return ConverterReturn(annotations, self.get_meta(), None)

    def get_meta(self):
        return {'no_recursion': True}
