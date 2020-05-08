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

from ..representation import QuestionAnsweringAnnotation
from ..utils import read_json
from ..config import PathField, NumberField, BoolField

from .format_converter import BaseFormatConverter, ConverterReturn
from ._nlp_common import SquadWordPieseTokenizer


class SQUADConverter(BaseFormatConverter):
    __provider__ = "squad"
    annotation_types = (QuestionAnsweringAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'testing_file': PathField(description="Path to testing file."),
            'vocab_file': PathField(description='Path to vocabulary file.'),
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
        self.tokenizer = SquadWordPieseTokenizer(
            self.get_value_from_config('vocab_file'), self.lower_case, max_len=self.max_seq_length
        )
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
                    is_impossible = qa.get('is_impossible', False)
                    example = {
                        'id': qas_id,
                        'question_text': question_text,
                        'context_text': paragraph_text,
                        'answer_text': orig_answer_text,
                        'char_to_word_offset': char_to_word_offset,
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

        for (example_index, example) in enumerate(examples):
            all_doc_tokens, tok_to_orig_index, _ = self.get_all_doc_tokens(example['tokens'])
            spans = []
            truncated_query = self.tokenizer.encode(
                example['question_text'], add_special_tokens=False, max_length=self.max_query_length
            )
            sequence_added_tokens = self.tokenizer.max_len - self.tokenizer.max_len_single_sentence
            sequence_pair_added_tokens = self.tokenizer.max_len - self.tokenizer.max_len_sentences_pair

            span_doc_tokens = all_doc_tokens
            while len(spans) * self.doc_stride < len(all_doc_tokens):
                encoded_dict = self.tokenizer.encode_plus(
                    truncated_query if self.tokenizer.padding_side == "right" else span_doc_tokens,
                    span_doc_tokens if self.tokenizer.padding_side == "right" else truncated_query,
                    max_length=self.max_seq_length,
                    return_overflowing_tokens=True,
                    pad_to_max_length=True,
                    stride=self.max_seq_length - self.doc_stride - len(truncated_query) - sequence_pair_added_tokens,
                    truncation_strategy="only_second" if self.tokenizer.padding_side == "right" else "only_first",
                )

                paragraph_len = min(
                    len(all_doc_tokens) - len(spans) * self.doc_stride,
                    self.max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
                )

                if self.tokenizer.pad_token_id in encoded_dict["input_ids"]:
                    non_padded_ids = encoded_dict["input_ids"][:encoded_dict["input_ids"].index(
                        self.tokenizer.pad_token_id
                    )]
                else:
                    non_padded_ids = encoded_dict["input_ids"]

                tokens = self.tokenizer.convert_ids_to_tokens(non_padded_ids)

                tr_q_len = len(truncated_query)
                encoded_dict["paragraph_len"] = paragraph_len
                encoded_dict["tokens"] = tokens
                encoded_dict["token_to_orig_map"] = self.get_token_to_orig_map(
                    paragraph_len, tr_q_len, sequence_added_tokens, spans, tok_to_orig_index
                )
                encoded_dict["truncated_query_with_special_tokens_length"] = tr_q_len + sequence_added_tokens
                encoded_dict["token_is_max_context"] = {}
                encoded_dict["start"] = len(spans) * self.doc_stride
                encoded_dict["length"] = paragraph_len

                spans.append(encoded_dict)

                if "overflowing_tokens" not in encoded_dict:
                    break
                span_doc_tokens = encoded_dict["overflowing_tokens"]

            self.set_max_context(spans)

            for span in spans:
                # Identify the position of the CLS token
                cls_index = span["input_ids"].index(self.tokenizer.cls_token_id)

                # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
                # Original TF implem also keep the classification token (set to 0)
                p_mask = np.array(span["token_type_ids"])

                p_mask = np.minimum(p_mask, 1)

                if self.tokenizer.padding_side == "right":
                    # Limit positive values to one
                    p_mask = 1 - p_mask

                p_mask[np.where(np.array(span["input_ids"]) == self.tokenizer.sep_token_id)[0]] = 1

                # Set the CLS index to '0'
                p_mask[cls_index] = 0
                idx = example_index
                identifier = ['input_ids_{}'.format(idx), 'input_mask_{}'.format(idx), 'segment_ids_{}'.format(idx)]

                annotation = QuestionAnsweringAnnotation(
                    identifier,
                    np.array(unique_id),
                    np.array(span["input_ids"]),
                    np.array(span["attention_mask"]),
                    np.array(span["token_type_ids"]),
                    np.array(cls_index),
                    p_mask,
                    answers[example_index],
                    example["context_text"],
                    example["tokens"],
                    example['is_impossible'],
                    span["paragraph_len"],
                    span["token_is_max_context"],
                    span["tokens"],
                    span["token_to_orig_map"],
                )
                annotation.metadata['lower_case'] = self.lower_case
                annotations.append(annotation)
                unique_id += 1

        return ConverterReturn(annotations, None, None)

    @staticmethod
    def _is_max_context(doc_spans, cur_span_index, position):
        """Check if this is the 'max context' doc span for the token."""
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span["start"] + doc_span["length"] - 1
            if position < doc_span["start"]:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span["start"]
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index


    def get_all_doc_tokens(self, tokens):
        all_doc_tokens = []
        tok_to_orig_index = []
        orig_to_tok_index = []
        for (i, token) in enumerate(tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = self.tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
        return all_doc_tokens, tok_to_orig_index, orig_to_tok_index

    def get_token_to_orig_map(self, paragraph_len, tr_q_len, sequence_added_tokens, spans, tok_to_orig_index):
        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = tr_q_len + sequence_added_tokens + i if self.tokenizer.padding_side == "right" else i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * self.doc_stride + i]
        return token_to_orig_map

    def set_max_context(self, spans):
        for doc_span_index, span in enumerate(spans):
            for j in range(span["paragraph_len"]):
                is_max_context = self._is_max_context(spans, doc_span_index, doc_span_index * self.doc_stride + j)
                index = (
                    j
                    if self.tokenizer.padding_side == "left"
                    else span["truncated_query_with_special_tokens_length"] + j
                )
                span["token_is_max_context"][index] = is_max_context
