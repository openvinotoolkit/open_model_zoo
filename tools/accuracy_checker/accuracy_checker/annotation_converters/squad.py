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

from ..representation import QuestionAnsweringAnnotation, ExtendedQuestionAnsweringAnnotation
from ..utils import read_json
from ..config import PathField, NumberField, BoolField

from .format_converter import BaseFormatConverter, ConverterReturn
from ._nlp_common import get_tokenizer, CLS_ID, SEP_ID

def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
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
            'lower_case': BoolField(optional=True, default=False, description='Switch tokens to lower case register'),
            'extended': BoolField(optional=True, default=False, description='Use extended mode tokenizer')
        })

        return configuration_parameters

    def configure(self):
        self.testing_file = self.get_value_from_config('testing_file')
        self.max_seq_length = int(self.get_value_from_config('max_seq_length'))
        self.max_query_length = self.get_value_from_config('max_query_length')
        self.doc_stride = self.get_value_from_config('doc_stride')
        self.lower_case = self.get_value_from_config('lower_case')
        self.extended = self.get_value_from_config('extended')
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
                    try:
                        is_impossible = qa['is_impossible']
                    except KeyError:
                        is_impossible = False

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

    # @staticmethod
    # def _extended_load_examples(file):
    #     def _is_whitespace(c):
    #         if c in [" ", "\t", "\r", "\n"] or ord(c) == 0x202F:
    #             return True
    #         return False
    #
    #     examples = []
    #     answers = []
    #     data = read_json(file)['data']
    #
    #     for entry in data:
    #         start_position, end_position = 0, 0
    #         doc_tokens = []
    #         char_to_word_offset = []
    #         prev_is_whitespace = True
    #
    #         # Split on whitespace so that different tokens may be attributed to their original position.
    #         for c in self.context_text:
    #             if _is_whitespace(c):
    #                 prev_is_whitespace = True
    #             else:
    #                 if prev_is_whitespace:
    #                     doc_tokens.append(c)
    #                 else:
    #                     doc_tokens[-1] += c
    #                 prev_is_whitespace = False
    #             char_to_word_offset.append(len(doc_tokens) - 1)
    #
    #         self.doc_tokens = doc_tokens
    #         self.char_to_word_offset = char_to_word_offset
    #
    #         # Start end end positions only has a value during evaluation.
    #         if start_position_character is not None and not is_impossible:
    #             self.start_position = char_to_word_offset[start_position_character]
    #             self.end_position = char_to_word_offset[
    #                 min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
    #             ]

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        if self.extended:
            annotations = self.convertExtended(check_content, progress_callback, progress_interval, **kwargs)
        else:
            annotations = self.convertBase(check_content, progress_callback, progress_interval, **kwargs)
        return ConverterReturn(annotations, None, None)

    def convertExtended(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):

        examples, answers = self._load_examples(self.testing_file)
        annotations = []
        unique_id = 1000000000
        DocSpan = namedtuple("DocSpan", ["start", "length"])

        for (example_index, example) in enumerate(examples):
            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(example['tokens']):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = self.tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

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
                    non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(self.tokenizer.pad_token_id)]
                else:
                    non_padded_ids = encoded_dict["input_ids"]

                tokens = self.tokenizer.convert_ids_to_tokens(non_padded_ids)

                token_to_orig_map = {}
                for i in range(paragraph_len):
                    index = len(truncated_query) + sequence_added_tokens + i if self.tokenizer.padding_side == "right" else i
                    token_to_orig_map[index] = tok_to_orig_index[len(spans) * self.doc_stride + i]

                encoded_dict["paragraph_len"] = paragraph_len
                encoded_dict["tokens"] = tokens
                encoded_dict["token_to_orig_map"] = token_to_orig_map
                encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
                encoded_dict["token_is_max_context"] = {}
                encoded_dict["start"] = len(spans) * self.doc_stride
                encoded_dict["length"] = paragraph_len

                spans.append(encoded_dict)

                if "overflowing_tokens" not in encoded_dict:
                    break
                span_doc_tokens = encoded_dict["overflowing_tokens"]

            for doc_span_index in range(len(spans)):
                for j in range(spans[doc_span_index]["paragraph_len"]):
                    is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * self.doc_stride + j)
                    index = (
                        j
                        if self.tokenizer.padding_side == "left"
                        else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
                    )
                    spans[doc_span_index]["token_is_max_context"][index] = is_max_context

            for span in spans:
                # Identify the position of the CLS token
                cls_index = span["input_ids"].index(self.tokenizer.cls_token_id)

                # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
                # Original TF implem also keep the classification token (set to 0) (not sure why...)
                p_mask = np.array(span["token_type_ids"])

                p_mask = np.minimum(p_mask, 1)

                if self.tokenizer.padding_side == "right":
                    # Limit positive values to one
                    p_mask = 1 - p_mask

                p_mask[np.where(np.array(span["input_ids"]) == self.tokenizer.sep_token_id)[0]] = 1

                # Set the CLS index to '0'
                p_mask[cls_index] = 0

                span_is_impossible = example['is_impossible']
                start_position = 0
                end_position = 0

                """
                Single squad example features to be fed to a model.
                Those features are model-specific and can be crafted from :class:`~transformers.data.processors.squad.SquadExample`
                using the :method:`~transformers.data.processors.squad.squad_convert_examples_to_features` method.

                Args:
                    input_ids: Indices of input sequence tokens in the vocabulary.
                    attention_mask: Mask to avoid performing attention on padding token indices.
                    token_type_ids: Segment token indices to indicate first and second portions of the inputs.
                    cls_index: the index of the CLS token.
                    p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
                        Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
                    example_index: the index of the example
                    unique_id: The unique Feature identifier
                    paragraph_len: The length of the context
                    token_is_max_context: List of booleans identifying which tokens have their maximum context in this feature object.
                        If a token does not have their maximum context in this feature object, it means that another feature object
                        has more information related to that token and should be prioritized over this feature for that token.
                    tokens: list of tokens corresponding to the input ids
                    token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
                    start_position: start of the answer token index
                    end_position: end of the answer token index
                """
                idx = example_index
                identifier = ['input_ids_{}'.format(idx), 'input_mask_{}'.format(idx), 'segment_ids_{}'.format(idx)]

                annotation = ExtendedQuestionAnsweringAnnotation(
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
                    start_position,
                    end_position
                )
                annotations.append(annotation)
                unique_id += 1

            # if (example_index + 1 ) % 101 == 0:
            #     break
            if example_index % 100 == 0:
                print(example_index)

        return annotations

    def convertBase(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):

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

                try:
                    for i in range(doc_span.length):
                        split_token_index = doc_span.start + i
                        tokens.append(all_doc_tokens[split_token_index])
                        segment_ids.append(1)
                except TypeError as e:
                    pass
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
        return annotations

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
