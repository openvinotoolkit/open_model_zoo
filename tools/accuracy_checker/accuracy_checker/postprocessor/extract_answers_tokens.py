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

from collections import OrderedDict
import numpy as np

from .postprocessor import Postprocessor
from ..representation import QuestionAnsweringAnnotation, QuestionAnsweringPrediction
from ..annotation_converters._nlp_common import WordPieceTokenizer
from ..config import NumberField


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
            indexes = np.argsort(logits)[::-1]
            score = np.array(logits)[indexes]
            best_indexes_mask = np.arange(len(score)) < n_best_size
            best_indexes = indexes[best_indexes_mask]
            return best_indexes

        def _extended_check_indexes(start, end, annotation, max_answer):
            if start >= len(annotation.tokens) or end >= len(annotation.tokens):
                return False
            if start not in annotation.token_to_orig_map or end not in annotation.token_to_orig_map:
                return False
            if not annotation.token_is_max_context.get(start, False):
                return False
            if end < start or end - start + 1 > max_answer:
                return False
            return True

        cnt = 0
        for annotation_, prediction_ in zip(annotation, prediction):
            start_indexes = _get_best_indexes(prediction_.start_logits, self.n_best_size)
            end_indexes = _get_best_indexes(prediction_.end_logits, self.n_best_size)
            valid_start_indexes = []
            valid_end_indexes = []
            tokens = []

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if _extended_check_indexes(start_index, end_index, annotation_, self.max_answer):
                        valid_start_indexes.append(start_index)
                        valid_end_indexes.append(end_index)
                        tokens.append(annotation_.tokens[start_index:(end_index + 1)])

            start_logits = prediction_.start_logits[valid_start_indexes]
            end_logits = prediction_.end_logits[valid_end_indexes]

            start_indexes = [val for _, val in sorted(zip(start_logits + end_logits, start_indexes), reverse=True)]
            if not start_indexes:
                continue
            start_indexes_ = start_indexes[0]
            end_indexes_ = [val for _, val in sorted(zip(start_logits + end_logits, end_indexes), reverse=True)]
            end_indexes_ = end_indexes_[0]

            prediction_.start_index.append(start_indexes_)
            prediction_.end_index.append(end_indexes_)

            tokens_ = [" ".join(tok) for _, tok in sorted(zip(start_logits + end_logits, tokens), reverse=True)]
            tokens_ = tokens_[0]
            tokens_ = tokens_.replace(" ##", "")
            tokens_ = tokens_.replace("##", "")
            tokens_ = tokens_.strip()

            if start_indexes_:
                tok_tokens = annotation_.tokens[start_indexes_:end_indexes_ + 1]
                orig_doc_start = annotation_.token_to_orig_map[start_indexes_]
                orig_doc_end = annotation_.token_to_orig_map[end_indexes_]
                orig_tokens = annotation_.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                tokens_ = self.get_final_text(tok_text, orig_text, annotation_.metadata.get('lower_case', False))

            prediction_.tokens.append(tokens_)
            cnt += 1

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
