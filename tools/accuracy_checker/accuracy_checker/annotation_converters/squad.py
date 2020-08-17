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

import numpy as np

from ..representation import QuestionAnsweringAnnotation
from ..utils import read_json
from ..config import PathField, NumberField, BoolField

from .format_converter import BaseFormatConverter, ConverterReturn
from ._nlp_common import SquadWordPieseTokenizer, _is_whitespace


def _check_is_max_context(doc_spans, cur_span_index, position):
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

class SquadExample:
    """
    A single training/test example for the Squad dataset, as loaded from disk.

    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
            self,
            qas_id,
            question_text,
            context_text,
            answer_text,
            start_position_character,
            title,
            answers=None,
            is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers or []

        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start and end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]

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
        self.max_seq_length = int(self.get_value_from_config('max_seq_length'))
        self.max_query_length = self.get_value_from_config('max_query_length')
        self.doc_stride = self.get_value_from_config('doc_stride')
        self.lower_case = self.get_value_from_config('lower_case')
        self.tokenizer = SquadWordPieseTokenizer(
            self.get_value_from_config('vocab_file'), self.lower_case, max_len=512
        )

    @staticmethod
    def _load_examples(file):
        examples = []
        data = read_json(file)['data']

        for entry in data:
            title = entry["title"]
            for paragraph in entry["paragraphs"]:
                context_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    answers = []

                    if "is_impossible" in qa:
                        is_impossible = qa["is_impossible"]
                    else:
                        is_impossible = False

                    if not is_impossible:
                        answers = qa["answers"]

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                    )

                    examples.append(example)
        return examples

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        examples = self._load_examples(self.testing_file)
        annotations = []
        unique_id = 1000000000

        for (example_index, example) in enumerate(examples):
            all_doc_tokens, tok_to_orig_index, _ = self.get_all_doc_tokens(example.doc_tokens)

            spans = []

            truncated_query = self.tokenizer.encode(
                example.question_text, add_special_tokens=False, max_length=self.max_query_length
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
                    return_token_type_ids=True,
                )

                paragraph_len = min(
                    len(all_doc_tokens) - len(spans) * self.doc_stride,
                    self.max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
                )

                if self.tokenizer.pad_token_id in encoded_dict["input_ids"]:
                    if self.tokenizer.padding_side == "right":
                        non_padded_ids = (
                            encoded_dict["input_ids"][:encoded_dict["input_ids"].index(self.tokenizer.pad_token_id)]
                        )
                    else:
                        pos = encoded_dict["input_ids"][::-1].index(self.tokenizer.pad_token_id)
                        last_padding_id_position = len(encoded_dict["input_ids"]) - 1 - pos
                        non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1:]

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

            spans = self.set_max_context(spans)

            for span in spans:
                # Identify the position of the CLS token
                cls_index = span["input_ids"].index(self.tokenizer.cls_token_id)

                p_mask = self.fill_p_mask(span, len(truncated_query), sequence_added_tokens, cls_index)
                idx = example_index
                identifier = ['input_ids_{}'.format(idx), 'input_mask_{}'.format(idx), 'segment_ids_{}'.format(idx)]
                annotation = QuestionAnsweringAnnotation(
                    identifier,
                    example.qas_id,
                    np.array(unique_id),
                    np.array(span["input_ids"]),
                    np.array(span["attention_mask"]),
                    np.array(span["token_type_ids"]),
                    np.array(cls_index),
                    p_mask,
                    example.answers,
                    example.context_text,
                    example.doc_tokens,
                    example.is_impossible,
                    span["paragraph_len"],
                    span["tokens"],
                    span["token_is_max_context"],
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
        return spans

    def fill_p_mask(self, span, tr_q_len, sequence_added_tokens, cls_index):
        p_mask = np.ones_like(span["token_type_ids"])
        if self.tokenizer.padding_side == "right":
            p_mask[tr_q_len + sequence_added_tokens:] = 0
        else:
            p_mask[-len(span["tokens"]): -(tr_q_len + sequence_added_tokens)] = 0

        pad_token_indices = np.where(span["input_ids"] == self.tokenizer.pad_token_id)
        special_token_indices = np.asarray(
            self.tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
        ).nonzero()

        p_mask[pad_token_indices] = 1
        p_mask[special_token_indices] = 1

        # Set the cls index to 0: the CLS index can be used for impossible answers
        p_mask[cls_index] = 0

        return p_mask
