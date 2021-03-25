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
from ..config import NumberField, StringField, ConfigError
from ..representation import DNASequencePrediction
from ..utils import UnsupportedPackage

try:
    from fast_ctc_decode import beam_search, viterbi_search
except ImportError as import_error:
    beam_search = UnsupportedPackage('fast_ctc_decode', import_error.msg)
    viterbi_search = UnsupportedPackage('fast_ctc_decode', import_error.msg)


class DNASeqRecognition(Adapter):
    __provider__ = 'dna_seq_beam_search'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'beam_size': NumberField(optional=True, value_type=int, default=5, min_value=1, description='size of beam'),
            'threshold': NumberField(
                optional=True, min_value=0, default=1e-3, description='threshold ofr valid prediction'
            ),
            'output_blob': StringField(optional=True, description='name of output layer')
        })
        return params

    def configure(self):
        if isinstance(beam_search, UnsupportedPackage):
            beam_search.raise_error(self.__provider__)
        self.beam_size = self.get_value_from_config('beam_size')
        self.threshold = self.get_value_from_config('threshold')
        self.output_blob = self.get_value_from_config('output_blob')

    def process(self, raw, identifiers, frame_meta):
        if not self.label_map:
            raise ConfigError('Beam Search Decoder requires dataset label map for correct decoding.')
        alphabet = list(self.label_map.values())
        raw_outputs = self._extract_predictions(raw, frame_meta)
        self.select_output_blob(raw_outputs)
        result = []
        for identifier, out in zip(identifiers, np.exp(raw_outputs[self.output_blob])):
            if self.beam_size == 1:
                seq, _ = viterbi_search(np.squeeze(out), alphabet, False, 1, 0)
            else:
                seq, _ = beam_search(np.squeeze(out.astype(np.float32)), alphabet, self.beam_size, self.threshold)
            result.append(DNASequencePrediction(identifier, seq))
        return result
