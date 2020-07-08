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
from collections import defaultdict
import string
import numpy as np

from ..adapters import Adapter
from ..config import ConfigValidator, NumberField, BoolField, ConfigError
from ..representation import CharacterRecognitionPrediction


class CTCBeamSearchDecoder(Adapter):
    __provider__ = 'ctc_beam_search_decoder'
    prediction_types = (CharacterRecognitionPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'beam_size': NumberField(
                optional=True, value_type=int, min_value=1, default=10,
                description="Size of the beam to use during decoding."
            ),
            'blank_label': NumberField(
                optional=True, value_type=int, min_value=0, description="Index of the CTC blank label."
            ),
            'softmaxed_probabilities': BoolField(
                optional=True, default=False, description="Indicator that model uses softmax for output layer "
            )
        })
        return parameters

    def configure(self):
        self.beam_size = self.get_value_from_config('beam_size')
        self.blank_label = self.launcher_config.get('blank_label')
        self.alphabet = ' ' + string.ascii_lowercase + '\'-'
        self.alphabet = self.alphabet.encode('ascii').decode('utf-8')


    def process(self, raw, identifiers=None, frame_meta=None):
        raw_output = self._extract_predictions(raw, frame_meta)
        output = raw_output[self.output_blob]
        output = np.swapaxes(output, 0, 1)

        result = []
        for identifier, data in zip(identifiers, output):
            if self.softmaxed_probabilities:
                data = np.log(data)
            seq = self.decode(data, self.beam_size, self.blank_label)
            decoded = ''.join(str(self.label_map[char]) for char in seq)
            result.append(CharacterRecognitionPrediction(identifier, decoded))
        return result

    @staticmethod
    def decode(probabilities, beamwidth=10, blank_id=None):
        pred = probabilities.squeeze()

        t_step = pred.shape[0]
        # idx_b = text_label.index(blank)
        idx_b = pred.shape[1] - 1

        _pB = {}
        _pNB = {}
        _pT = {}

        _init = ()  # init state, to make sure the first index is not blank ****

        for __t in ['c', 'l']:
            _pB[__t] = {}
            _pNB[__t] = {}
            _pT[__t] = {}

        _pB['l'][_init] = 1
        _pNB['l'][_init] = 0
        _pT['l'][_init] = 1

        for _t in range(t_step):
            _pB['c'] = {}
            _pNB['c'] = {}
            _pT['c'] = {}

            for _candidate in _pNB['l']:
                _TpNB = 0
                if _candidate != _init:
                    _TpNB = _pNB['l'][_candidate] * pred[_t][_candidate[-1]]
                _TpB = _pT['l'][_candidate] * pred[_t][idx_b]
                if _candidate in _pNB['c']:
                    _pNB['c'][_candidate] += _TpNB
                else:
                    _pNB['c'][_candidate] = _TpNB
                _pB['c'][_candidate] = _TpB
                _pT['c'][_candidate] = _pNB['c'][_candidate] + _pB['c'][_candidate]

                for i, v in np.ndenumerate(pred[_t]):
                    if i < (idx_b,):
                        extand_t = _candidate + (i,)
                        if len(_candidate) > 0 and _candidate[-1] == i:
                            _TpNB = v * _pB['l'][_candidate]

                        else:
                            _TpNB = v * _pT['l'][_candidate]

                        if extand_t in _pT['c']:
                            _pT['c'][extand_t] += _TpNB
                            _pNB['c'][extand_t] += _TpNB
                        else:
                            _pB['c'][extand_t] = 0
                            _pT['c'][extand_t] = _TpNB
                            _pNB['c'][extand_t] = _TpNB

            sorted_c = sorted(_pT['c'].items(), reverse=True, key=lambda item: item[1])
            _pB['l'] = {}
            _pNB['l'] = {}
            _pT['l'] = {}
            for _sent in sorted_c[:beamwidth]:
                _pB['l'][_sent[0]] = _pB['c'][_sent[0]]
                _pNB['l'][_sent[0]] = _pNB['c'][_sent[0]]
                _pT['l'][_sent[0]] = _pT['c'][_sent[0]]

        res = sorted(_pT['l'].items(), reverse=True, key=lambda item: item[1])[0]
        decoded = ''.join([self.alphabet[t[0]] for t in res[0]])

        return decoded


