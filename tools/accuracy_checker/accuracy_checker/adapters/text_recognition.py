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

from collections import defaultdict
import numpy as np

from ..adapters import Adapter
from ..config import ConfigValidator, ConfigError, NumberField, BoolField, DictField, ListField, StringField, PathField
from ..representation import CharacterRecognitionPrediction
from ..utils import softmax


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
            ),
            'logits_output': StringField(optional=True, description='Logits output layer name'),
            'custom_label_map': DictField(optional=True, description='Label map')
        })
        return parameters

    @classmethod
    def validate_config(cls, config, fetch_only=False, **kwargs):
        return super().validate_config(
            config, fetch_only=fetch_only, on_extra_argument=ConfigValidator.IGNORE_ON_EXTRA_ARGUMENT
        )

    def configure(self):
        self.beam_size = self.get_value_from_config('beam_size')
        self.blank_label = self.launcher_config.get('blank_label')
        self.softmaxed_probabilities = self.get_value_from_config('softmaxed_probabilities')
        self.logits_output = self.get_value_from_config("logits_output")
        self.custom_label_map = self.get_value_from_config("custom_label_map")
        if self.custom_label_map:
            labels = {int(k): v for k, v in self.custom_label_map.items()}
            self.custom_label_map = labels

    def process(self, raw, identifiers, frame_meta):
        if self.custom_label_map:
            self.label_map = self.custom_label_map
        if not self.label_map:
            raise ConfigError('Beam Search Decoder requires dataset label map for correct decoding.')
        if self.blank_label is None:
            self.blank_label = len(self.label_map)
        if self.logits_output:
            self.output_blob = self.logits_output
        raw_output = self._extract_predictions(raw, frame_meta)
        self.select_output_blob(raw_output)
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
        beam = [((), (0.0, -np.inf))]

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
            ),
            'logits_output': StringField(optional=True, description='Logits output layer name'),
            'custom_label_map': DictField(optional=True, description='Label map')

        })
        return parameters

    @classmethod
    def validate_config(cls, config, fetch_only=False, **kwargs):
        return super().validate_config(
            config, fetch_only=fetch_only, on_extra_argument=ConfigValidator.IGNORE_ON_EXTRA_ARGUMENT
        )

    def configure(self):
        self.blank_label = self.get_value_from_config('blank_label')
        self.logits_output = self.get_value_from_config("logits_output")
        self.custom_label_map = self.get_value_from_config("custom_label_map")
        if self.custom_label_map:
            labels = {int(k): v for k, v in self.custom_label_map.items()}
            self.custom_label_map = labels

    def process(self, raw, identifiers=None, frame_meta=None):
        if self.custom_label_map:
            self.label_map = self.custom_label_map
        if not self.label_map:
            raise ConfigError('CTCGreedy Search Decoder requires dataset label map for correct decoding.')
        if self.blank_label is None:
            self.blank_label = 0
        if self.logits_output:
            self.output_blob = self.logits_output
        raw_output = self._extract_predictions(raw, frame_meta)
        self.select_output_blob(raw_output)
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


class SimpleDecoder(Adapter):
    __provider__ = 'simple_decoder'
    prediction_types = (CharacterRecognitionPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'eos_label': StringField(
                optional=True, default='[s]', description="End-of-sequence label."
            ),
            'custom_label_map': DictField(optional=True, description='Label map')

        })
        return parameters

    @classmethod
    def validate_config(cls, config, fetch_only=False, **kwargs):
        return super().validate_config(
            config, fetch_only=fetch_only, on_extra_argument=ConfigValidator.IGNORE_ON_EXTRA_ARGUMENT
        )

    def configure(self):
        self.eos_label = self.get_value_from_config('eos_label')
        self.custom_label_map = self.get_value_from_config("custom_label_map")
        if self.custom_label_map:
            labels = {int(k): v for k, v in self.custom_label_map.items()}
            self.custom_label_map = labels

    def process(self, raw, identifiers=None, frame_meta=None):
        if self.custom_label_map:
            self.label_map = self.custom_label_map
        if not self.label_map:
            raise ConfigError('Decoder requires dataset label map for correct decoding.')
        raw_output = self._extract_predictions(raw, frame_meta)
        self.select_output_blob(raw_output)
        output = raw_output[self.output_blob]
        output = softmax(output, 2)
        preds_index = np.argmax(output, 2)

        result = []
        for identifier, data in zip(identifiers, preds_index):
            decoded = ''.join(str(self.label_map[char]) for char in data)
            decoded = decoded[:decoded.find(self.eos_label)]
            result.append(CharacterRecognitionPrediction(identifier, decoded))

        return result


class LPRAdapter(Adapter):
    __provider__ = 'lpr'
    prediction_types = (CharacterRecognitionPrediction,)

    def process(self, raw, identifiers=None, frame_meta=None):
        if not self.label_map:
            raise ConfigError('LPR adapter requires dataset label map for correct decoding.')
        raw_output = self._extract_predictions(raw, frame_meta)
        self.select_output_blob(raw_output)
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


class AttentionOCRAdapter(Adapter):
    __provider__ = 'aocr'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'output_blob': StringField(description='network output with predicted labels name', optional=True),
            'labels': ListField(
                description='label list for decoding', optional=True,
                default=['', '', ''] + [chr(i) for i in range(32, 127)]),
            'eos_index': NumberField(
                default=2, optional=True, description='end of string symbol index', value_type=int),
            'to_lower_case': BoolField(optional=True, default=True,
                                       description='should be output string converted to lower case or not')
        })
        return params

    def configure(self):
        self.output_blob = self.get_value_from_config('output_blob')
        self.labels = self.get_value_from_config('labels')
        self.eos_index = self.get_value_from_config('eos_index')
        self.lower_case = self.get_value_from_config('to_lower_case')

    def process(self, raw, identifiers, frame_meta):
        raw_out = self._extract_predictions(raw, frame_meta)
        self.select_output_blob(raw_out)
        result = []
        if isinstance(raw_out[self.output_blob], bytes):
            out_str = raw_out[self.output_blob].decode('iso-8859-1')
            if self.lower_case:
                out_str = out_str.lower()
            return [CharacterRecognitionPrediction(identifiers[0], out_str)]

        for identifier, out in zip(identifiers, raw_out[self.output_blob]):
            valid_out = out[out != self.eos_index]
            decoded_out = ''.join([self.labels[idx] for idx in valid_out])
            if self.lower_case:
                decoded_out = decoded_out.lower()
            result.append(CharacterRecognitionPrediction(identifier, decoded_out))
        return result


class PDPDTextRecognition(Adapter):
    __provider__ = 'ppocr'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'vocabulary_file': PathField(description='file with decoding labels'),
            'remove_duplicates': BoolField(
                optional=True, default=True, description='remove duplications from result string'
            )
        })
        return params

    def configure(self):
        self.labels_file = self.get_value_from_config('vocabulary_file')
        chr_str = ''
        with self.labels_file.open("rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                chr_str += line
            chr_str += " "
        dict_character = list(chr_str)
        self.label_map = dict(enumerate(['<blank>'] + dict_character))
        self.ignored_tokens = [0]
        self.remove_duplicates = self.get_value_from_config('remove_duplicates')

    def process(self, raw, identifiers, frame_meta):
        results = []
        outputs = self._extract_predictions(raw, frame_meta)
        for identifier, out in zip(identifiers, outputs[self.output_blob]):
            preds_idx = np.argmax(out, axis=1)
            text = self.decode(preds_idx)
            results.append(CharacterRecognitionPrediction(identifier, text))

        return results

    def decode(self, text_index):
        """ convert text-index into text-label. """
        char_list = []
        for pos_id, idx in enumerate(text_index):
            if idx in self.ignored_tokens:
                continue
            if self.remove_duplicates:
                if pos_id > 0 and text_index[pos_id - 1] == idx:
                    continue

            if idx == len(self.label_map):
                continue
            char_list.append(self.label_map[int(idx)])

        return ''.join(char_list)
