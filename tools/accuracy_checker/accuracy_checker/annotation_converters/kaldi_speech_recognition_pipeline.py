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
from ..data_readers import KaldiMatrixIdentifier, KaldiARKReader
from ..representation import CharacterRecognitionAnnotation
from ..config import PathField, BoolField
from ..utils import read_txt
from .format_converter import BaseFormatConverter, ConverterReturn


class KaldiSpeechRecognitionDataConverter(BaseFormatConverter):
    __provider__ = 'kaldi_asr_data'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'annotation_file': PathField(description='file with gt transcription'),
            'data_dir': PathField(description='directory with ark files', is_directory=True),
            'features_subset_file': PathField(description='file with list testing ark files', optional=True),
            'ivectors': BoolField(optional=True, default=False, description='include ivectors features')
        })
        return params

    def configure(self):
        self.annotation_file = self.get_value_from_config('annotation_file')
        self.data_dir = self.get_value_from_config('data_dir')
        self.feat_list_file = self.get_value_from_config('features_subset_file')
        self.ivectors = self.get_value_from_config('ivectors')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        ark_list = self.select_subset()
        transcripts = self.read_annotation()
        annotations = []
        for ark in ark_list:
            ivect = None
            if isinstance(ark, tuple):
                ark, ivect = ark
            utterances = KaldiARKReader.read_frames(ark)
            for utt in utterances:
                if utt not in transcripts:
                    continue

                identifier = (
                    KaldiMatrixIdentifier(ark.name, utt)
                    if not ivect else [KaldiMatrixIdentifier(ark.name, utt), KaldiMatrixIdentifier(ivect.name, utt)]
                )

                gt = transcripts[utt]
                annotations.append(CharacterRecognitionAnnotation(identifier, gt))

        return ConverterReturn(annotations, None, None)

    def select_subset(self):
        if not self.ivectors:
            if self.feat_list_file:
                return [self.data_dir / ark for ark in read_txt(self.feat_list_file)]
            return list(self.data_dir.glob('*.ark'))
        if self.feat_list_file:
            return [
                (self.data_dir / ark.split(' ')[0], self.data_dir / ark.split(' ')[1])
                for ark in read_txt(self.feat_list_file)
            ]
        pairs = []
        for ivector_file in self.data_dir.glob("*_ivector.ark"):
            feats_file = self.data_dir / ivector_file.name.replace('_ivector', '')
            if not feats_file.exists():
                continue
            pairs.append((feats_file, ivector_file))
        return pairs

    def read_annotation(self):
        trascript_dict = {}
        for line in read_txt(self.annotation_file):
            utterance_key, text = line.split(' ', 1)
            trascript_dict[utterance_key] = text
        return trascript_dict
