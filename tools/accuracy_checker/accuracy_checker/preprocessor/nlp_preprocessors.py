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

from .preprocessor import Preprocessor
from ..config import PathField, NumberField, StringField, BoolField, ConfigError
from ..utils import read_txt, UnsupportedPackage

try:
    from tokenizers import SentencePieceBPETokenizer
except ImportError as import_error:
    SentencePieceBPETokenizer = UnsupportedPackage("tokenizers", import_error.msg)

class DecodeBySentencePieceBPETokenizer(Preprocessor):
    __provider__ = 'decode_by_sentence_piece_bpe_tokenizer'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'vocabulary_file': PathField(),
            'merges_file': PathField(),
            'sos_symbol': StringField(optional=True, default='<s>'),
            'eos_symbol': StringField(optional=True, default='</s>'),
            'add_extra_symbols': BoolField(optional=True, default=True),
        })

        return parameters

    def configure(self):
        if isinstance(SentencePieceBPETokenizer, UnsupportedPackage):
            SentencePieceBPETokenizer.raise_error(self.__provider__)
        self.tokenizer = SentencePieceBPETokenizer(
            str(self.get_value_from_config('vocabulary_file')),
            str(self.get_value_from_config('merges_file'))
        )
        self.add_extra_symbols = self.get_value_from_config('add_extra_symbols')
        self.idx = {}
        for s in ['sos', 'eos']:
            self.idx[s] = self.tokenizer.token_to_id(
                str(self.get_value_from_config(s + '_symbol'))
            )

    def process(self, image, annotation_meta=None):
        sentence = " ".join(image.data)
        tokens = self.tokenizer.encode(sentence).ids
        if self.add_extra_symbols:
            tokens = [self.idx['sos']] + tokens + [self.idx['eos']]
        image.data = tokens
        image.metadata['decoded'] = True
        image.identifier = "tokens"

        return image


class DecodeByVocabulary(Preprocessor):
    __provider__ = 'decode_by_vocabulary'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'vocabulary_file': PathField(),
            'unk_index': NumberField(value_type=int, min_value=0),
        })

        return parameters

    def configure(self):
        vocab = read_txt(self.get_value_from_config('vocabulary_file'), encoding='utf-8')
        decoding_dict = {}
        for idx, word in enumerate(vocab):
            decoding_dict[word] = idx
        self.unk_index = self.get_value_from_config('unk_index')
        self.decoding_dict = decoding_dict

    def process(self, image, annotation_meta=None):
        sentence = image.data
        words = sentence.split(' ') if isinstance(sentence, str) else sentence
        decoded_sentence = []
        for word in words:
            decoded_sentence.append(self.decoding_dict.get(word, self.unk_index))
        image.data = decoded_sentence
        image.metadata['decoded'] = True

        return image


class PadWithEOS(Preprocessor):
    __provider__ = 'pad_with_eos'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update(
            {
                'eos_index': NumberField(value_type=int, min_value=0, optional=True),
                'eos_symbol': StringField(optional=True),
                'sequence_len': NumberField(value_type=int, min_value=1)
            }
        )

        return parameters

    def configure(self):
        self.eos_id = self.get_value_from_config('eos_index')
        self.eos_symbol = self.get_value_from_config('eos_symbol')
        self.sequence_len = self.get_value_from_config('sequence_len')

        if self.eos_id is None and self.eos_symbol is None:
            raise ConfigError('eos_index or eos_symbol should be provided')

    def process(self, image, annotation_meta=None):
        data = image.data
        decoded = image.metadata.get('decoded', False)
        if decoded:
            if self.eos_id is None:
                raise ConfigError('eos_index should be specified')
            if len(data) >= self.sequence_len:
                image.data = data[:self.sequence_len]
                return image
            meaningful_data_len = len(data)
            addition_size = self.sequence_len - meaningful_data_len
            data.extend([self.eos_id] * addition_size)
            image.data = data
            return image

        if self.eos_symbol is None:
            raise ConfigError('eos_symbol should be specified')
        words = data.split(' ')
        if len(words) >= self.sequence_len:
            words = words[:self.sequence_len]
            image.data = ' '.join(words)
            return image
        meaningful_data_len = len(words)
        addition_size = self.sequence_len - meaningful_data_len
        addition = [self.eos_symbol] * addition_size
        words.extend(addition)
        image.data = ' '.join(words)

        return image
