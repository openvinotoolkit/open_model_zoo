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

import numpy as np
from .format_converter import BaseFormatConverter, ConverterReturn
from ..config import PathField, BoolField, NumberField
from ..utils import read_txt
from ..representation import LM1BAnnotation
from ..data_readers import MultiFramesInputIdentifier


class LM1BDatasetConverter(BaseFormatConverter):
    __provider__ = 'lm_1b'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update(
            {
                'input_file': PathField(description='Input file for word prediction'),
                'vocab_file': PathField(description='model vocabulary file'),
                'chars_encoding': BoolField(
                    optional=True, default=False, description='include encoding words by chars'
                ),
                'max_word_length': NumberField(optional=True, value_type=int, default=50, min_value=1),
                'multi_inputs': NumberField(optional=True, value_type=int, default=2, min_value=1)
            }
        )
        return params

    def configure(self):
        self.input_file = self.get_value_from_config('input_file')
        self.chars_encoding = self.get_value_from_config('chars_encoding')
        self.max_word_length = self.get_value_from_config('max_word_length')
        self.load_vocab(self.get_value_from_config('vocab_file'))
        self.multi_inputs = self.get_value_from_config('multi_inputs')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        sentences = read_txt(self.input_file)
        encoded_sentences = [self.encode_sentence(sentence) for sentence in sentences]
        encoded_by_chars_sentences = []
        unique_y = []
        unique_idx = []
        if self.chars_encoding:
            encoded_by_chars_sentences = [self.encode_by_chars(sentence_ids) for sentence_ids in encoded_sentences]
        if self.chars_encoding:
            for sentence_id, sentence in enumerate(encoded_sentences):
                encoded_y = []
                encoded_idx = []
                encoded_by_chars_sentence = encoded_by_chars_sentences[sentence_id]
                for word_idx in range(len(encoded_by_chars_sentence)):
                    encoded_char = np.asarray(encoded_by_chars_sentence[word_idx])
                    _y, _idx = self.encode_unique_by_chars(encoded_char)
                    encoded_y.append(_y)
                    encoded_idx.append(_idx)
                unique_y.append(encoded_y)
                unique_idx.append(encoded_idx)
        annotations = []
        num_iters = len(encoded_sentences)
        unique_input_ids = list(range(self.multi_inputs))
        for sentence_id, sentence in enumerate(encoded_sentences):
            if progress_callback and sentence_id % progress_interval == 0:
                progress_callback(sentence_id / num_iters * 100)
            targets = sentence[1:]
            unique_input_identifier = ['sentence_{}_{}'.format(sentence_id, unique_ids) for unique_ids in unique_input_ids]
            annotations.append(
                LM1BAnnotation(
                    MultiFramesInputIdentifier(unique_input_ids, unique_input_identifier), sentence, targets,
                    encoded_by_chars_sentences[sentence_id] if self.chars_encoding else None,
                    unique_y[sentence_id] if self.chars_encoding else None,
                    unique_idx[sentence_id] if self.chars_encoding else None)
            )

        return ConverterReturn(
            annotations, {'label_map': self.vocab, 'unk': self._unk, 'bos': self._bos, 'eos': self._eos}, None
        )

    def load_vocab(self, vocab_file):
        words = read_txt(vocab_file)
        self.word_to_id = {}
        id_to_word = []
        idx = 0
        for word in words:
            if word == '<S>':
                self._bos = idx
            if word == '</S>':
                self._eos = idx
            if word == '<UNK>':
                self._unk = idx
            if word == '!!!MAXTERMID':
                continue
            id_to_word.append(word)
            self.word_to_id[word] = idx
            idx += 1
        self.vocab = dict(enumerate(id_to_word))

        if self.chars_encoding:
            chars_set = set()
            for word in id_to_word:
                chars_set |= set(word)
            free_ids = [chr(i) for i in range(256) if chr(i) not in chars_set]
            if len(free_ids) < 5:
                raise ValueError('chars encoding impossible, not enough free chars for specific symbols ids')
            self._bos_char, self._eos_char, self._bow_char, self._eow_char, self._pad_char = free_ids[:5]
            chars_set |= {self._bos_char, self._eos_char, self._bow_char, self._eow_char, self._pad_char}
            self._word_to_char_ids = []
            for word in id_to_word:
                padded_chars = np.full(self.max_word_length, ord(self._pad_char))
                if len(word) > self.max_word_length - 2:
                    word = word[:self.max_word_length - 2]
                padded_chars[0] = ord(self._bow_char)
                for char_id, char in enumerate(word):
                    padded_chars[char_id + 1] = ord(char)
                padded_chars[len(word) + 1] = ord(self._eow_char)
                self._word_to_char_ids.append(padded_chars)
            self._bos_char_ids = np.full(self.max_word_length, ord(self._pad_char))
            self._bos_char_ids[:3] = [ord(self._bow_char), ord(self._bos_char), ord(self._eow_char)]
            self._eos_char_ids = np.full(self.max_word_length, ord(self._pad_char))
            self._eos_char_ids[:3] = [ord(self._bow_char), ord(self._eos_char), ord(self._eow_char)]

    def encode_sentence(self, sentence):
        words = sentence.split()
        encoded_sentence = [self._bos]
        word_ids = [self.word_to_id.get(word, self._unk) for word in words]
        encoded_sentence.extend(word_ids)
        encoded_sentence.append(self._eos)

        return encoded_sentence

    def encode_by_chars(self, encoded_sentence):
        sentence_rep = [self._bos_char_ids]
        for word_id in encoded_sentence[1:-1]:
            sentence_rep.append(self._word_to_char_ids[word_id])
        return sentence_rep

    @staticmethod
    def encode_unique_by_chars(char_input):
        def _unique_(x):
            y_ = []
            idx_ = []
            for xi in x:
                if xi not in y_:
                    y_.append(xi)
            for xi in x:
                for index, result in enumerate(xi == y_):
                    if result:
                        idx_.append(index)
                        break
            return y_, idx_

        x = char_input.reshape(np.array([-1]))
        y, idx = _unique_(x)
        shape = len(x)
        if len(y) >= shape:
            return y, idx
        _y = []
        _y.extend(y)
        for _ in range(shape-len(y)):
            _y.append(y[len(y)-1])
        return _y, idx
