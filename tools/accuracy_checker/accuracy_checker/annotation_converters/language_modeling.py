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

from .format_converter import BaseFormatConverter, ConverterReturn
from ..config import PathField
from ..utils import read_txt
from ..representation import LMAnnotation


class LanguageModelDatasetConverter(BaseFormatConverter):
    __provider__ = 'lm_1b'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update(
            {
                'input_file': PathField(description='Input file for word prediction'),
                'vocab_file': PathField(description='model vocabulary file')
            }
        )
        return params

    def configure(self):
        self.input_file = self.get_value_from_config('input_file')
        self.load_vocab(self.get_value_from_config('vocab_file'))

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        sentences = read_txt(self.input_file)
        encoded_sentences = [self.encode_sentence(sentence) for sentence in sentences]
        annotations = []
        for sentence_id, sentence in enumerate(encoded_sentences):
            targets = sentence[1:-1]
            annotations.append(LMAnnotation('sentence_{}'.format(sentence_id), sentence, targets))

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
            id_to_word.append(idx)
            self.word_to_id[word] = idx
            idx += 1
        self.vocab = dict(enumerate(id_to_word))

    def encode_sentence(self, sentence):
        words = sentence.split()
        encoded_sentence = [self._bos]
        word_ids = [self.word_to_id.get(word, self._unk) for word in words]
        encoded_sentence.extend(word_ids)
        encoded_sentence.append(self._eos)

        return encoded_sentence
