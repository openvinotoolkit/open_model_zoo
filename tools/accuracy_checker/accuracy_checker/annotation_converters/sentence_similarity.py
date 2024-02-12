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

import csv
from collections import namedtuple
import numpy as np
from ..config import NumberField, BoolField, PathField, StringField, ConfigError
from ..representation import SentenceSimilarityAnnotation
from .format_converter import FileBasedAnnotationConverter, ConverterReturn
from ._nlp_common import get_tokenizer, SEG_ID_A, SEP_ID, CLS_ID, SEG_ID_CLS, SEG_ID_PAD


input_sample = namedtuple('input_sample', ['id', 'text', 'pair_id', 'score'])


class SentenceSimilarityConverter(FileBasedAnnotationConverter):
    __provider__ = 'sentence_similarity'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'annotation_file': PathField(description='path to annotation file in json or tsv format'),
            'vocab_file': PathField(description='Path to vocabulary file for word piece tokenizer', optional=True),
            'sentence_piece_model_file': PathField(description='sentence piece model for tokenization', optional=True),
            'max_seq_length': NumberField(
                description='The maximum total input sequence length after tokenization.',
                optional=True, default=128, value_type=int
            ),
            'lower_case': BoolField(optional=True, default=False, description='Switch tokens to lower case register'),
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
            'column_separator': StringField(
                optional=True, choices=['tab', 'comma'], description='column separator used in annotation file'
            ),
            'text_1': NumberField(value_type=int, description='Column index for text', optional=True, default=0),
            'text_2': NumberField(value_type=int, description='Second column index for text', optional=True, default=1),
            'label': NumberField(value_type=int, description='Label column', optional=True, default=2),
            'normalize_labels': BoolField(
                optional=True, default=False, description='normalize similarity score to have 0, 1 range'
            )
        })

        return params

    def configure(self):
        self.annotation_file = self.get_value_from_config('annotation_file')
        self.text_a_id = self.get_value_from_config('text_1')
        self.text_b_id = self.get_value_from_config('text_2')
        self.label_id = self.get_value_from_config('label')
        self.column_separator = self.get_column_separator()
        self.max_seq_length = self.get_value_from_config('max_seq_length')
        self.lower_case = self.get_value_from_config('lower_case')
        self.tokenizer, self.external_tok = get_tokenizer(self.config, self.lower_case)
        self.support_vocab = 'vocab_file' in self.config
        self.class_token_first = self.get_value_from_config('class_token_first')
        self.enable_padding = self.get_value_from_config('enable_padding')
        self.normalize_labels = self.get_value_from_config('normalize_labels')

    def get_column_separator(self):
        sep = self.get_value_from_config('column_separator')
        if sep is None:
            if self.annotation_file.suffix not in ['.csv', '.tsv']:
                raise ConfigError(
                    'Impossible automatically detect column separator for annotation. '
                    'Please provide separator in config')
            sep = 'comma' if self.annotation_file.suffix == '.csv' else 'tab'
        return ',' if sep == 'comma' else '\t'

    def read_annotation(self):
        lines = []
        all_pair_scores = []
        with open(str(self.annotation_file), 'r', encoding="utf-8-sig") as ann_file:
            reader = csv.reader(ann_file, delimiter=self.column_separator, quotechar=None)
            for idx, line in enumerate(reader):
                if idx == 0:
                    continue
                first_idx = idx * 2
                second_idx = first_idx + 1
                score = float(line[self.label_id])
                all_pair_scores.append(score)
                text_a = line[self.text_a_id]
                text_b = line[self.text_b_id]
                lines.append(input_sample(first_idx, text_a, second_idx, score))
                lines.append(input_sample(second_idx, text_b, None, None))

        return lines, all_pair_scores

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        examples, scores = self.read_annotation()
        min_score, max_score = None, None
        if self.normalize_labels:
            min_score = min(scores)
            max_score = max(scores)
        annotations = []
        num_iter = len(examples)
        for example_id, example in enumerate(examples):
            annotations.append(self.convert_single_example(example, min_score, max_score))
            if progress_callback and example_id % progress_interval == 0:
                progress_callback(example_id * 100 / num_iter)

        return ConverterReturn(annotations, None, None)

    def convert_single_example(self, example, min_score=None, max_score=None):
        identifier = [
            'input_ids_{}'.format(example.id),
            'input_mask_{}'.format(example.id),
            'segment_ids_{}'.format(example.id)
        ]
        if not self.external_tok:
            tokens_a = self.tokenizer.tokenize(example.text)
            if len(tokens_a) > self.max_seq_length - 2:
                tokens_a = tokens_a[:self.max_seq_length - 2]

            tokens = []
            segment_ids = []
            if self.class_token_first:
                tokens.append("[CLS]" if self.support_vocab else CLS_ID)
                segment_ids.append(SEG_ID_CLS)
            for token in tokens_a:
                tokens.append(token)
                segment_ids.append(SEG_ID_A)
            tokens.append('[SEP]' if self.support_vocab else SEP_ID)
            segment_ids.append(SEG_ID_A)

            if not self.class_token_first:
                tokens.append("[CLS]" if self.support_vocab else CLS_ID)
                segment_ids.append(SEG_ID_CLS)
        else:
            tokens = self.tokenizer.tokenize(example.text, add_special_tokens=True)
            segment_ids = [SEG_ID_A] * len(tokens)

            if len(tokens) > self.max_seq_length:
                tokens = tokens[:self.max_seq_length]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens) if self.support_vocab or self.external_tok else tokens
        input_mask = [0 if not self.class_token_first else 1] * len(input_ids)

        if self.enable_padding and len(input_ids) < self.max_seq_length:
            delta_len = self.max_seq_length - len(input_ids)
            input_ids = [0] * delta_len + input_ids if not self.class_token_first else input_ids + [0] * delta_len
            input_mask = [1] * delta_len + input_mask if not self.class_token_first else input_mask + [0] * delta_len
            segment_ids = (
                [SEG_ID_PAD] * delta_len + segment_ids if not self.class_token_first else segment_ids + [0] * delta_len
            )
        score = example.score
        if max_score is not None and score is not None  :
            score = (score - min_score) / max_score

        return SentenceSimilarityAnnotation(
            identifier, example.id, example.pair_id, score,
            np.array(input_ids), np.array(input_mask), np.array(segment_ids)
        )
