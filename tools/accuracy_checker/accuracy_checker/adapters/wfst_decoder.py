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

import numpy as np
from .adapter import Adapter
from ..config import PathField, BoolField, NumberField
from ..utils import read_txt, UnsupportedPackage
from ..representation import CharacterRecognitionPrediction

try:
    from kaldi.asr import MappedLatticeFasterRecognizer
    from kaldi.matrix import Matrix
    from kaldi.decoder import LatticeFasterDecoderOptions
except ImportError as err:
    MappedLatticeFasterRecognizer = UnsupportedPackage('pykaldi', err.msg)
    Matrix = UnsupportedPackage('pykaldi', err.msg)
    LatticeFasterDecoderOptions = UnsupportedPackage('pykaldi', err.msg)


class WFSTDecodingAdapter(Adapter):
    __provider__ = 'kaldi_lattice_decoder'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'fst_file': PathField(description='WFST state graph file'),
            'words_file': PathField(description='words table file'),
            'transition_model_file': PathField(description='transition model file'),
            'beam': NumberField(optional=True, value_type=int, min_value=1, description='beam size'),
            'lattice_beam': NumberField(optional=True, value_type=int, min_value=1, description='lattice beam size'),
            'allow_partial': BoolField(optional=True, default=True, description='allow decoding'),
            'acoustic_scale': NumberField(optional=True, default=0.1, value_type=float),
            'min_active': NumberField(optional=True, value_type=int, min_value=9),
            'max_active': NumberField(optional=True, value_type=int, min_value=0),
        })
        return params

    def configure(self):
        if isinstance(MappedLatticeFasterRecognizer, UnsupportedPackage):
            MappedLatticeFasterRecognizer.raise_error(self.__provider__)
        self.fst_file = self.get_value_from_config('fst_file')
        self.words_file = self.get_value_from_config('words_file')
        self.transition_model = self.get_value_from_config('transition_model_file')
        self.words_table = self.read_words_table()
        decoder_opts = LatticeFasterDecoderOptions()
        beam = self.get_value_from_config('beam')
        if beam:
            decoder_opts.beam = beam
        lattice_beam = self.get_value_from_config('lattice_beam')
        if lattice_beam:
            decoder_opts.lattice_beam = lattice_beam
        min_active = self.get_value_from_config('min_active')
        max_active = self.get_value_from_config('max_active')
        if min_active:
            decoder_opts.min_active = min_active
        if max_active:
            decoder_opts.max_active = max_active

        self._decoder = MappedLatticeFasterRecognizer.from_files(
            str(self.transition_model), str(self.fst_file), str(self.words_file),
            allow_partial=self.get_value_from_config('allow_partial'),
            acoustic_scale=self.get_value_from_config('acoustic_scale'),
            decoder_opts=decoder_opts,
        )

    def process(self, raw, identifiers, frame_meta):
        results = []
        preds = self._extract_predictions(raw, frame_meta)
        for identifier, log_scores in zip(identifiers, preds[self.output_blob]):
            seq = self._decoder.decode(Matrix(log_scores))
            trans = seq['text']
            results.append(CharacterRecognitionPrediction(identifier, trans))
        return results

    def read_words_table(self):
        words_table = {}
        for line in read_txt(self.words_file):
            word, idx = line.split()
            words_table[int(idx)] = word
        return words_table

    def _extract_predictions(self, outputs_list, meta):
        is_multi_infer = meta[-1].get('multi_infer', False) if meta else False
        if not is_multi_infer:
            return outputs_list[0] if not isinstance(outputs_list, dict) else outputs_list

        output_map = {
            self.output_blob: np.expand_dims(np.concatenate([out[self.output_blob] for out in outputs_list], axis=0), 0)
        }

        return output_map
