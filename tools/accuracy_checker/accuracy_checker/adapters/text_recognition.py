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

from collections import defaultdict

import numpy as np

from ..adapters import Adapter
from ..config import ConfigValidator, NumberField, BoolField, ConfigError
from ..representation import CharacterRecognitionPrediction


class BeamSearchDecoder(Adapter):
    __provider__ = 'beam_search_decoder'
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

    def validate_config(self):
        super().validate_config(on_extra_argument=ConfigValidator.IGNORE_ON_EXTRA_ARGUMENT)
        self.beam_size = self.get_value_from_config('beam_size')
        self.blank_label = self.launcher_config.get('blank_label')
        self.softmaxed_probabilities = self.get_value_from_config('softmaxed_probabilities')

    def process(self, raw, identifiers, frame_meta):
        if not self.label_map:
            raise ConfigError('Beam Search Decoder requires dataset label map for correct decoding.')
        if self.blank_label is None:
            self.blank_label = len(self.label_map)
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
    def decode(probabilities, beam_size=10, blank_id=None):
        """
         Decode given output probabilities to sequence of labels.
        Arguments:
            probabilities: The output log probabilities for each time step.
            Should be an array of shape (time x output dim).
            beam_size (int): Size of the beam to use during decoding.
            blank_id (int): Index of the CTC blank label.
        Returns the output label sequence.
        """
        def make_new_beam():
            return defaultdict(lambda: (-np.inf, -np.inf))

        def log_sum_exp(*args):
            if all(a == -np.inf for a in args):
                return -np.inf
            a_max = np.max(args)
            lsp = np.log(sum(np.exp(a - a_max) for a in args))

            return a_max + lsp

        times, symbols = probabilities.shape
        # Initialize the beam with the empty sequence, a probability of 1 for ending in blank
        # and zero for ending in non-blank (in log space).
        beam = [(tuple(), (0.0, -np.inf))]

        for time in range(times):
            # A default dictionary to store the next step candidates.
            next_beam = make_new_beam()

            for symbol_id in range(symbols):
                current_prob = probabilities[time, symbol_id]

                for prefix, (prob_blank, prob_non_blank) in beam:
                    # If propose a blank the prefix doesn't change.
                    # Only the probability of ending in blank gets updated.
                    if symbol_id == blank_id:
                        next_prob_blank, next_prob_non_blank = next_beam[prefix]
                        next_prob_blank = log_sum_exp(
                            next_prob_blank, prob_blank + current_prob, prob_non_blank + current_prob
                        )
                        next_beam[prefix] = (next_prob_blank, next_prob_non_blank)
                        continue
                    # Extend the prefix by the new character symbol and add it to the beam.
                    # Only the probability of not ending in blank gets updated.
                    end_t = prefix[-1] if prefix else None
                    next_prefix = prefix + (symbol_id,)
                    next_prob_blank, next_prob_non_blank = next_beam[next_prefix]
                    if symbol_id != end_t:
                        next_prob_non_blank = log_sum_exp(
                            next_prob_non_blank, prob_blank + current_prob, prob_non_blank + current_prob
                        )
                    else:
                        # Don't include the previous probability of not ending in blank (prob_non_blank) if symbol
                        #  is repeated at the end. The CTC algorithm merges characters not separated by a blank.
                        next_prob_non_blank = log_sum_exp(next_prob_non_blank, prob_blank + current_prob)

                    next_beam[next_prefix] = (next_prob_blank, next_prob_non_blank)
                    # If symbol is repeated at the end also update the unchanged prefix. This is the merging case.
                    if symbol_id == end_t:
                        next_prob_blank, next_prob_non_blank = next_beam[prefix]
                        next_prob_non_blank = log_sum_exp(next_prob_non_blank, prob_non_blank + current_prob)
                        next_beam[prefix] = (next_prob_blank, next_prob_non_blank)

            beam = sorted(next_beam.items(), key=lambda x: log_sum_exp(*x[1]), reverse=True)[:beam_size]
        best = beam[0]
        return best[0]


class CTCGreedySearchDecoder(Adapter):
    __provider__ = 'ctc_greedy_search_decoder'
    prediction_types = (CharacterRecognitionPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'blank_label': NumberField(
                optional=True, value_type=int, min_value=0, default=0, description="Index of the CTC blank label."
            )
        })
        return parameters

    def validate_config(self):
        super().validate_config(on_extra_argument=ConfigValidator.IGNORE_ON_EXTRA_ARGUMENT)
        self.blank_label = self.launcher_config.get('blank_label')

    def process(self, raw, identifiers=None, frame_meta=None):
        if not self.label_map:
            raise ConfigError('CTCGreedy Search Decoder requires dataset label map for correct decoding.')
        if self.blank_label is None:
            self.blank_label = 0
        raw_output = self._extract_predictions(raw, frame_meta)
        output = raw_output[self.output_blob]
        preds_index = np.argmax(output, 2)
        preds_index = preds_index.transpose(1, 0)

        result = []
        for identifier, data in zip(identifiers, preds_index):
            seq = self.decode(data, self.blank_label)
            decoded = ''.join(str(self.label_map[char]) for char in seq)
            result.append(CharacterRecognitionPrediction(identifier, decoded))

        return result

    @staticmethod
    def decode(prob_index, blank_id):
        """
         Decode given output probabilities to sequence of labels.
        Arguments:
            prob_index: The max index along the probabilities dimension.
            blank_id (int): Index of the CTC blank label.
        Returns the output label sequence.
        """
        index_length = prob_index.shape[0]
        selected_index = []
        for i in range(index_length):
            # removing repeated characters and blank.
            if prob_index[i] != blank_id and (not (i > blank_id and prob_index[i - 1] == prob_index[i])):
                selected_index.append(prob_index[i])
        return selected_index


class LPRAdapter(Adapter):
    __provider__ = 'lpr'
    prediction_types = (CharacterRecognitionPrediction,)

    def process(self, raw, identifiers=None, frame_meta=None):
        if not self.label_map:
            raise ConfigError('LPR adapter requires dataset label map for correct decoding.')
        raw_output = self._extract_predictions(raw, frame_meta)
        predictions = raw_output[self.output_blob]
        result = []
        for identifier, output in zip(identifiers, predictions):
            decoded_out = self.decode(output.reshape(-1))
            result.append(CharacterRecognitionPrediction(identifier, decoded_out))

        return result

    def decode(self, outputs):
        decode_out = str()
        for output in outputs:
            if output == -1:
                break
            decode_out += str(self.label_map[int(output)])

        return decode_out
