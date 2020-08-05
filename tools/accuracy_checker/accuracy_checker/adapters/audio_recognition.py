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
import string
import numpy as np

from ..adapters import Adapter
from ..config import NumberField, BoolField, StringField, ListField
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
            ),
            'classification_out': StringField(
                optional=True, default=None, description="Name of output node"
            )

        })
        return parameters

    def configure(self):
        self.beam_size = self.get_value_from_config('beam_size')
        self.blank_label = self.launcher_config.get('blank_label')
        self.softmaxed_probabilities = self.launcher_config.get('softmaxed_probabilities')
        self.classification_out = self.get_value_from_config('classification_out')
        self.alphabet = ' ' + string.ascii_lowercase + '\'-'
        self.alphabet = self.alphabet.encode('ascii').decode('utf-8')

    def process(self, raw, identifiers=None, frame_meta=None):
        if self.classification_out is not None:
            self.output_blob = self.classification_out
        multi_infer = frame_meta[-1].get('multi_infer', False) if frame_meta else False

        raw_output = self._extract_predictions(raw, frame_meta)
        output = raw_output[self.output_blob]
        if multi_infer:
            steps, _, _, _ = output.shape
            res = []
            for i in range(steps):
                res.append(output[i, ...])
            output = np.concatenate(tuple(res))

        result = []
        if self.softmaxed_probabilities:
            output = np.log(output)
        seq = self.decode(output, self.beam_size, self.blank_label)
        decoded = ''.join([self.alphabet[t[0]] for t in seq[0]])
        decoded = decoded.upper()
        result.append(CharacterRecognitionPrediction(identifiers[0], decoded))
        return result

    @staticmethod
    def _extract_predictions(outputs_list, meta):
        is_multi_infer = meta[-1].get('multi_infer', False) if meta else False
        if not is_multi_infer:
            return outputs_list[0] if not isinstance(outputs_list, dict) else outputs_list

        output_map = {}
        for output_key in outputs_list[0].keys():
            output_data = np.asarray([output[output_key] for output in outputs_list])
            output_map[output_key] = output_data

        return output_map

    @staticmethod
    def decode(probabilities, beamwidth=10, blank_id=None):
        pred = probabilities.squeeze()

        t_step = pred.shape[0]
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

            for _cddt in _pNB['l']:
                _TpNB = 0
                if _cddt != _init:
                    _TpNB = _pNB['l'][_cddt] * pred[_t][_cddt[-1]]
                _TpB = _pT['l'][_cddt] * pred[_t][idx_b]

                _pNB['c'][_cddt] = _TpNB + _pNB['c'][_cddt] if _cddt in _pNB['c'] else _TpNB

                _pB['c'][_cddt] = _TpB
                _pT['c'][_cddt] = _pNB['c'][_cddt] + _pB['c'][_cddt]

                nonblanks = [(i, v) for i, v in np.ndenumerate(pred[_t]) if i < (idx_b,)]
                for nb in nonblanks:
                    i, v = nb

                    extand_t = _cddt + (i,)
                    _TpNB = v * _pB['l'][_cddt] if len(_cddt) > 0 and _cddt[-1] == i else v * _pT['l'][_cddt]

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

        return res


class CTCGreedyDecoder(Adapter):
    __provider__ = 'ctc_greedy_decoder'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'alphabet': ListField(optional=True),
            'softmaxed_probabilities': BoolField(
                optional=True, default=False, description="Indicator that model uses softmax for output layer "
            ),
            'classification_out': StringField(
                optional=True, default=None, description="Name of output node"
            )
        })
        return params

    def configure(self):
        self.alphabet = self.get_value_from_config('alphabet') or ' ' + string.ascii_lowercase + '\'-'
        self.softmaxed_probabilities = self.launcher_config.get('softmaxed_probabilities')
        self.classification_out = self.get_value_from_config('classification_out')

    @staticmethod
    def _extract_predictions(outputs_list, meta):
        is_multi_infer = meta[-1].get('multi_infer', False) if meta else False
        if not is_multi_infer:
            return outputs_list[0] if not isinstance(outputs_list, dict) else outputs_list

        output_map = {}
        for output_key in outputs_list[0].keys():
            output_data = np.asarray([output[output_key] for output in outputs_list])
            output_map[output_key] = output_data

        return output_map

    def process(self, raw, identifiers, frame_meta):
        if self.classification_out is not None:
            self.output_blob = self.classification_out
        multi_infer = frame_meta[-1].get('multi_infer', False) if frame_meta else False

        raw_output = self._extract_predictions(raw, frame_meta)
        output = raw_output[self.output_blob]
        if multi_infer:
            steps, _, _, _ = output.shape
            res = []
            for i in range(steps):
                res.append(output[i, ...])
            output = np.concatenate(tuple(res))

        if self.softmaxed_probabilities:
            output = np.log(output)
        argmx = output.argmax(axis=-1)

        decoded = self._ctc_decoder_prediction(argmx, self.alphabet)[0].upper()
        return [CharacterRecognitionPrediction(identifiers[0], decoded)]

    @staticmethod
    def _ctc_decoder_prediction(prediction, labels):
        """
        Decodes a sequence of labels to words
        """
        blank_id = len(labels)
        hypotheses = []
        # CTC decoding procedure
        for ind in range(prediction.shape[0]):
            decoded_prediction = []
            previous = blank_id
            pr = prediction[ind]
            for p in pr:
                if (p != previous or previous == blank_id) and p != blank_id:
                    decoded_prediction.append(p)
                previous = p
            hypothesis = ''.join([labels[c] for c in decoded_prediction])
            hypotheses.append(hypothesis)
        return hypotheses
