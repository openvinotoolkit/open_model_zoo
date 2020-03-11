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

from collections import namedtuple

import numpy as np

from ..representation import QuestionAnsweringAnnotation
from ..utils import read_json
from ..config import PathField, NumberField, BoolField

from .format_converter import BaseFormatConverter, ConverterReturn
from ._nlp_common import get_tokenizer, CLS_ID, SEP_ID


class SQUADConverter(BaseFormatConverter):
    __provider__ = "squad"
    annotation_types = (QuestionAnsweringAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'testing_file': PathField(description="Path to testing file."),
            'vocab_file': PathField(description='Path to vocabulary file.', optional=True),
            'sentence_piece_model_file': PathField(description='sentence piece model for tokenization', optional=True),
            'max_seq_length': NumberField(
                description='The maximum total input sequence length after WordPiece tokenization.',
                optional=True, default=128, value_type=int
            ),
            'max_query_length': NumberField(
                description='The maximum number of tokens for the question.',
                optional=True, default=64, value_type=int
            ),
            'doc_stride': NumberField(
                description="When splitting up a long document into chunks, how much stride to take between chunks.",
                optional=True, default=128, value_type=int
            ),
            'lower_case': BoolField(optional=True, default=False, description='Switch tokens to lower case register')
        })

        return configuration_parameters

    def configure(self):
        self.testing_file = self.get_value_from_config('testing_file')
        self.max_seq_length = self.get_value_from_config('max_seq_length')
        self.max_query_length = self.get_value_from_config('max_query_length')
        self.doc_stride = self.get_value_from_config('doc_stride')
        self.lower_case = self.get_value_from_config('lower_case')
        self.tokenizer = get_tokenizer(self.config, self.lower_case)
        self.support_vocab = 'vocab_file' in self.config

    @staticmethod
    def _load_examples(file):
        def _is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        examples = []
        answers = []
        data = read_json(file)['data']

        for entry in data:
            for paragraph in entry['paragraphs']:
                paragraph_text = paragraph["context"]
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if _is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    orig_answer_text = qa["answers"]
                    is_impossible = False

                    example = {
                        'id': qas_id,
                        'question_text': question_text,
                        'tokens': doc_tokens,
                        'is_impossible': is_impossible
                    }
                    examples.append(example)
                    answers.append(orig_answer_text)
        return examples, answers

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        examples, answers = self._load_examples(self.testing_file)
        annotations = []
        unique_id = 1000000000
        DocSpan = namedtuple("DocSpan", ["start", "length"])

        for (example_index, example) in enumerate(examples):
            query_tokens = self.tokenizer.tokenize(example['question_text'])
            if len(query_tokens) > self.max_query_length:
                query_tokens = query_tokens[:self.max_query_length]
            all_doc_tokens = []
            for (i, token) in enumerate(example['tokens']):
                sub_tokens = self.tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    all_doc_tokens.append(sub_token)
            max_tokens_for_doc = self.max_seq_length - len(query_tokens) - 3
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(DocSpan(start_offset, length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, self.doc_stride)

            for idx, doc_span in enumerate(doc_spans):
                tokens = []
                segment_ids = []
                tokens.append("[CLS]" if self.support_vocab else CLS_ID)
                segment_ids.append(0)
                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append("[SEP]" if self.support_vocab else SEP_ID)
                segment_ids.append(0)

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(1)
                tokens.append("[SEP]" if self.support_vocab else SEP_ID)
                segment_ids.append(1)
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens) if self.support_vocab else tokens
                input_mask = [1] * len(input_ids)

                while len(input_ids) < self.max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                # add index to make identifier unique
                identifier = ['input_ids_{}'.format(idx), 'input_mask_{}'.format(idx), 'segment_ids_{}'.format(idx)]
                annotation = QuestionAnsweringAnnotation(
                    identifier,
                    np.array(unique_id),
                    np.array(input_ids),
                    np.array(input_mask),
                    np.array(segment_ids),
                    tokens,
                    answers[example_index],
                )
                annotations.append(annotation)
                unique_id += 1
        return ConverterReturn(annotations, None, None)

    @staticmethod
    def _is_max_context(doc_spans, cur_span_index, position):
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index
