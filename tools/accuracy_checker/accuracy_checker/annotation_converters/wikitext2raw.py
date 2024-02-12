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

from ..representation import LanguageModelingAnnotation
from ..config import PathField, NumberField, StringField, BoolField
from ..utils import UnsupportedPackage
from .format_converter import BaseFormatConverter, ConverterReturn
from ._nlp_common import get_tokenizer

try:
    from tokenizers import Tokenizer, pre_tokenizers, decoders
    from tokenizers.models import BPE
except ImportError as import_error:
    Tokenizer = UnsupportedPackage("tokenizers", import_error.msg)
    pre_tokenizers = UnsupportedPackage("tokenizers", import_error.msg)
    decoders = UnsupportedPackage("tokenizers", import_error.msg)
    BPE = UnsupportedPackage("tokenizers.models", import_error.msg)


class Wikitext2RawConverter(BaseFormatConverter):
    __provider__ = "wikitext2raw"
    annotation_types = (LanguageModelingAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'testing_file': PathField(description="Path to testing file."),
            'merges_file': PathField(description="Path to merges file."),
            'vocab_file': PathField(description='Path to vocabulary file.'),
            'class_token_first': BoolField(
                optional=True, default=True,
                description='Add [CLS] token to the begin of sequence. If False, will be added as the last token.'),
            'enable_padding': BoolField(optional=True, default=True, description='pad input sequence to max length'),
            'tokenizer_dir': PathField(
                optional=True, is_directory=True,
                description='A path to a directory containing vocabulary files required by the transformers tokenizer'
            ),
            'model_id': StringField(
                optional=True,
                description='The model id of a predefined tokenizer hosted inside a model repo on huggingface.co'
            ),
            'lower_case': BoolField(optional=True, default=False, description='converts output to lower case'),
            'max_seq_length': NumberField(
                description='The maximum total input sequence length after tokenization.',
                optional=True, default=128, value_type=int
            ),
        })

        return configuration_parameters

    def configure(self):
        if isinstance(Tokenizer, UnsupportedPackage):
            Tokenizer.raise_error(self.__provider__)
        self.testing_file = self.get_value_from_config('testing_file')
        self.vocab_file = self.get_value_from_config('vocab_file')
        self.merges_file = self.get_value_from_config('merges_file')
        self.max_seq_length = int(self.get_value_from_config('max_seq_length'))
        self.model_id = self.get_value_from_config('model_id')
        self.lower_case = self.get_value_from_config('lower_case')
        self.tokenizer, self.external_tok = get_tokenizer(self.config, self.lower_case)
        if not self.external_tok:
            self.tokenizer = Tokenizer(BPE.from_file(str(self.vocab_file), str(self.merges_file)))
            self.tokenizer.decoder = decoders.ByteLevel()
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        with open(str(self.testing_file), encoding="utf-8") as f:
            text = f.read()

        tokens = self.tokenizer([text]) if self.external_tok else self.tokenizer.encode_batch([text])

        encoding = tokens[0]
        annotations = []
        unique_id = 1000000000
        for idx in range(0, len(encoding.ids) - self.max_seq_length + 1, self.max_seq_length):
            ids = encoding.ids[idx: idx + self.max_seq_length]
            tokens = encoding.tokens[idx:idx + self.max_seq_length]
            attention_mask = encoding.attention_mask[idx:idx + self.max_seq_length]
            identifier = ['input_ids_{}'.format(idx), 'input_mask_{}'.format(idx), 'labels_{}'.format(idx)]
            annotation = LanguageModelingAnnotation(
                identifier,
                np.array(unique_id),
                np.array([ids]),
                tokens,
                labels=np.array(ids),
                input_mask=np.array([attention_mask])
            )
            annotations.append(annotation)
            unique_id += 1

        return ConverterReturn(annotations, None, None)
