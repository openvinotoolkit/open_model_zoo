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

from functools import singledispatch
from collections import OrderedDict, namedtuple
from .postprocessor import Postprocessor
from ..representation import (
    QuestionAnsweringAnnotation, QuestionAnsweringPrediction,
    QuestionAnsweringBiDAFAnnotation
)
from ..annotation_converters._nlp_common import WordPieceTokenizer
from ..config import NumberField


PrelimPrediction = namedtuple(
        "PrelimPrediction", ["start_index", "end_index", "start_logit", "end_logit", 'tokens']
    )

class ExtractSQUADPrediction(Postprocessor):
    """
    Extract text answers from predictions
    """

    __provider__ = 'extract_answers_tokens'

    annotation_types = (QuestionAnsweringAnnotation, )
    prediction_types = (QuestionAnsweringPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'max_answer': NumberField(
                optional=True, value_type=int, default=30, description="Maximum length of answer"
            ),
            'n_best_size': NumberField(
                optional=True, value_type=int, default=20, description="The total number of n-best predictions."
            )
        })
        return parameters

    def configure(self):
        self.max_answer = self.get_value_from_config('max_answer')
        self.n_best_size = self.get_value_from_config('n_best_size')

    def process_image(self, annotation, prediction):
        def _get_best_indexes(logits, n_best_size):
            """Get the n-best logits from a list."""
            index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

            best_indexes = []
            for i, _ in enumerate(index_and_score):
                if i >= n_best_size:
                    break
                best_indexes.append(index_and_score[i][0])
            return best_indexes

        def _extended_check_indexes(start, end, annotation, max_answer):
            if start >= len(annotation.tokens) or end >= len(annotation.tokens):
                return False
            if start not in annotation.token_to_orig_map or end not in annotation.token_to_orig_map:
                return False
            if not annotation.token_is_max_context.get(start, False):
                return False
            if end < start:
                return False
            if end - start + 1 > max_answer:
                return False
            return True


        for annotation_, prediction_ in zip(annotation, prediction):
            start_indexes = _get_best_indexes(prediction_.start_logits, self.n_best_size)
            end_indexes = _get_best_indexes(prediction_.end_logits, self.n_best_size)
            prelim_predictions = []

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if _extended_check_indexes(start_index, end_index, annotation_, self.max_answer):
                        prelim_predictions.append(
                            PrelimPrediction(
                                start_index,
                                end_index,
                                prediction_.start_logits[start_index],
                                prediction_.end_logits[end_index],
                                annotation_.tokens[start_index:(end_index + 1)]
                            )
                        )

            prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)
            nbest = []
            seen_predictions = set()
            for pred in prelim_predictions:
                if len(nbest) >= self.n_best_size:
                    break

                if pred.start_index > 0:
                    orig_doc_start = annotation_.token_to_orig_map[pred.start_index]
                    orig_doc_end = annotation_.token_to_orig_map[pred.end_index]
                    orig_tokens = annotation_.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                    tok_text = " ".join(pred.tokens).replace(" ##", "").strip()
                    # Clean whitespace
                    tok_text = tok_text.strip()
                    tok_text = " ".join(tok_text.split())
                    orig_text = " ".join(orig_tokens)

                    tokens_ = self.get_final_text(tok_text, orig_text, annotation_.metadata.get('lower_case', False))
                    if tokens_ in seen_predictions:
                        continue
                    nbest.append(tokens_)
                    seen_predictions.add(tokens_)
                else:
                    nbest.append('')
                    seen_predictions.add('')

            if not nbest:
                nbest.append("")

            prediction_.tokens = nbest

        return annotation, prediction

    @staticmethod
    def get_final_text(pred_text, orig_text, do_lower_case):
        def _strip_spaces(text):
            ns_chars = []
            ns_to_s_map = OrderedDict()
            for (i, c) in enumerate(text):
                if c == " ":
                    continue
                ns_to_s_map[len(ns_chars)] = i
                ns_chars.append(c)
            ns_text = "".join(ns_chars)
            return (ns_text, ns_to_s_map)

        tok_text = " ".join(WordPieceTokenizer.basic_tokenizer(orig_text, lower_case=do_lower_case))

        start_position = tok_text.find(pred_text)
        if start_position == -1:
            return orig_text
        end_position = start_position + len(pred_text) - 1

        (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
        (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

        if len(orig_ns_text) != len(tok_ns_text):
            return orig_text

        # We then project the characters in `pred_text` back to `orig_text` using
        # the character-to-character alignment.
        tok_s_to_ns_map = {}
        for (i, tok_index) in tok_ns_to_s_map.items():
            tok_s_to_ns_map[tok_index] = i

        orig_start_position = None
        if start_position in tok_s_to_ns_map:
            ns_start_position = tok_s_to_ns_map[start_position]
            if ns_start_position in orig_ns_to_s_map:
                orig_start_position = orig_ns_to_s_map[ns_start_position]

        if orig_start_position is None:
            return orig_text

        orig_end_position = None
        if end_position in tok_s_to_ns_map:
            ns_end_position = tok_s_to_ns_map[end_position]
            if ns_end_position in orig_ns_to_s_map:
                orig_end_position = orig_ns_to_s_map[ns_end_position]

        if orig_end_position is None:
            return orig_text

        output_text = orig_text[orig_start_position:(orig_end_position + 1)]
        return output_text

class ExtractSQUADPredictionBiDAF(Postprocessor):

    __provider__ = 'bidaf_extract_answers_tokens'

    annotation_types = (QuestionAnsweringBiDAFAnnotation, QuestionAnsweringAnnotation, )
    prediction_types = (QuestionAnsweringPrediction, )

    def process_image(self, annotation, prediction):
        def _extended_check_indexes(start, end, context_word, words_idx_in_context):
            if end >= len(context_word):
                return False
            if start >= len(words_idx_in_context) or end >= len(words_idx_in_context):
                return False
            if end < start:
                return False
            return True

        def _get_text(raw_text, indexes, start, end, end_length):
            return raw_text[indexes[start]:indexes[end] + end_length]

        @singledispatch
        def get_annotation_params(entry):
            return None, None, None

        @get_annotation_params.register(QuestionAnsweringBiDAFAnnotation)
        def _(entry):
            context = entry.context
            context_word = list(entry.context_word.reshape(-1))
            words_idx_in_context = entry.words_idx_in_context

            return context, context_word, words_idx_in_context

        @get_annotation_params.register(QuestionAnsweringAnnotation)
        def _(entry):
            def _get_tokens_indexes_in_context(context, words):
                indexes = []
                rem = context.lower()
                offset = 0
                for w in words:
                    idx = rem.find(w)
                    assert idx >= 0
                    indexes.append(idx + offset)
                    offset += idx + len(w)
                    rem = rem[idx + len(w):]
                return indexes

            context = entry.paragraph_text
            context_word = entry.tokens
            words_idx_in_context = _get_tokens_indexes_in_context(context, context_word)

            return context, context_word, words_idx_in_context


        for annotation_, prediction_ in zip(annotation, prediction):
            start_index = prediction_.start_index
            end_index = prediction_.end_index
            context, context_word, words_idx_in_context = get_annotation_params(annotation_)
            tokens = []
            if _extended_check_indexes(start_index, end_index, context_word, words_idx_in_context):
                end_length = len(context_word[end_index])
                tok_text = _get_text(context, words_idx_in_context,
                                     start_index, end_index, end_length)
                tokens.append(tok_text)
            else:
                tokens.append("")
            prediction_.tokens = tokens

        return annotation, prediction
