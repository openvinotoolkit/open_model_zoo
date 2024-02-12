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
import numpy as np
from ..data_readers import KaldiMatrixIdentifier, KaldiARKReader, KaldiFrameIdentifier
from ..representation import CharacterRecognitionAnnotation, RegressionAnnotation
from ..config import PathField, BoolField, StringField
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
        for ivector_file in self.data_dir.glob("*_ivector*.ark"):
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


class KaldiFeatureRegressionConverter(BaseFormatConverter):
    __provider__ = 'kaldi_feat_regression'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'data_dir': PathField(description='directory with ark files', is_directory=True),
            'ref_data_dir': PathField(description='directory with ref data', is_directory=True, optional=True),
            'features_subset_file': PathField(description='file with list testing ark files', optional=True),
            'ivectors': BoolField(optional=True, default=False, description='include ivectors features'),
            'ref_file_suffix': StringField(optional=True, default='_kaldi_score'),
            'vectors_mode': BoolField(optional=True, default=True, description='Split data to vectors'),
            'utterance_name_agnostic': BoolField(
                optional=True, default=False, description='do not match names per utterance'
            ),
            'use_numpy_data': BoolField(
                optional=True, default=False, description='allow to search npz files instead of ark'
            )
        })
        return params

    def configure(self):
        self.data_dir = self.get_value_from_config('data_dir')
        self.feat_list_file = self.get_value_from_config('features_subset_file')
        self.ivectors = self.get_value_from_config('ivectors')
        self.ref_data_dir = self.get_value_from_config('ref_data_dir')
        if self.ref_data_dir is None:
            self.ref_data_dir = self.data_dir
        self.ref_file_suffix = self.get_value_from_config('ref_file_suffix')
        self.vectors_mode = self.get_value_from_config('vectors_mode')
        self.utt_agnostic = self.get_value_from_config('utterance_name_agnostic')
        self.file_ext = '.ark' if not self.get_value_from_config('use_numpy_data') else '.npz'

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        ark_list = self.select_subset()
        annotation = []
        for files in ark_list:
            input_files, ref_ark = files[:-1], files[-1]
            if not self.utt_agnostic:
                annotation = self._convert_utt_specific(input_files, ref_ark, annotation)
            else:
                annotation = self._convert_utt_agnostic(input_files, ref_ark, annotation)

        return ConverterReturn(annotation, None, None)

    def _convert_utt_agnostic(self, input_files, ref_ark, annotation):
        input_utts = []
        for in_file in input_files:
            input_utts.append(KaldiARKReader.read_frames(in_file) if in_file.suffix != '.npz' else np.load(in_file))
        utt_ids = [list(in_utt.keys()) for in_utt in input_utts]
        ref_scores = KaldiARKReader.read_frames(ref_ark) if ref_ark.suffix != '.npz' else np.load(ref_ark)
        for idx, (_, ref_matrix) in enumerate(ref_scores.items()):
            current_utts = [u[idx] for u in utt_ids]
            if self.vectors_mode:
                for v_idx, ref_v in enumerate(ref_matrix):
                    identifier = [
                            KaldiFrameIdentifier(in_file.name, utt, v_idx)
                            if in_file.suffix != '.npz' else generate_numpy_identifier(in_file.name, utt, v_idx)
                            for in_file, utt in zip(input_files, current_utts)
                    ]
                    if len(identifier) == 1:
                        identifier = identifier[0]
                    annotation.append(RegressionAnnotation(identifier, ref_v))
            else:
                identifier = [
                        KaldiMatrixIdentifier(in_file.name, utt)
                        if in_file.suffix != '.npz' else generate_numpy_identifier(in_file.name, utt)
                        for in_file, utt in zip(input_files, current_utts)
                ]
                if len(identifier) == 1:
                    identifier = identifier[0]
                annotation.append(RegressionAnnotation(identifier, ref_matrix))
        return annotation

    def _convert_utt_specific(self, input_files, ref_ark, annotation):
        utterances = (
            KaldiARKReader.read_frames(input_files[0])
            if input_files[0].suffix != '.npz' else dict(np.load(input_files[0])))
        ref_scores = KaldiARKReader.read_frames(ref_ark) if ref_ark.suffix != '.npz' else dict(np.load(ref_ark))
        for utt, matrix in utterances.items():
            if utt not in ref_scores:
                continue
            ref_matrix = ref_scores[utt]
            if self.vectors_mode:
                for vector_id, _ in enumerate(matrix):
                    identifier = [
                            KaldiFrameIdentifier(in_file.name, utt, vector_id)
                            if in_file.suffix != '.npz' else generate_numpy_identifier(in_file.name, utt, vector_id)
                            for in_file in input_files
                    ]
                    if len(identifier) == 1:
                        identifier = identifier[0]
                    ref_vector = ref_matrix[vector_id]
                    annotation.append(RegressionAnnotation(identifier, ref_vector))
            else:
                identifier = [KaldiMatrixIdentifier(in_file.name, utt)
                              if in_file.suffix != '.npz' else generate_numpy_identifier(in_file.name, utt)
                              for in_file in input_files]
                if len(identifier) == 1:
                    identifier = identifier[0]
                annotation.append(RegressionAnnotation(identifier, ref_matrix))
        return annotation

    def select_subset(self):
        if self.feat_list_file:
            subset = []
            for ark in read_txt(self.feat_list_file):
                files = [self.data_dir / f for f in ark.split(' ')[:-1]]
                files.append(self.ref_data_dir / ark.split(' ')[-1])
                subset.append(files)
            return subset

        if not self.ivectors:
            pairs = []
            for ark_file in self.data_dir.glob('*{}'.format(self.file_ext)):
                if self.data_dir == self.ref_data_dir and self.ref_file_suffix in ark_file.name:
                    continue
                ref_file = self.ref_data_dir / ark_file.name.replace(self.file_ext, self.ref_file_suffix+self.file_ext)
                pairs.append((ark_file, ref_file))
            return pairs
        triples = []
        for ivector_file in self.data_dir.glob("*_ivector{}".format(self.file_ext)):
            feats_file = self.data_dir / ivector_file.name.replace('_ivector', '')
            ref_file = self.ref_data_dir / ivector_file.name.replace('_ivector', self.ref_file_suffix)
            if not feats_file.exists() or not ref_file.exists():
                continue
            triples.append((feats_file, ivector_file, ref_file))
        return triples


def generate_numpy_identifier(file_name, array_id, idx=None):
    return '{}{}#{}'.format(array_id, '' if idx is None else '_{}'.format(idx), file_name)
