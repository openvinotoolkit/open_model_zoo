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

from .postprocessor import Postprocessor
from ..representation import QuestionAnsweringAnnotation, QuestionAnsweringPrediction
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

        def _check_indexes(start, end, length, max_answer):
            if start >= length or end >= length:
                return False
            if end < start or end - start + 1 > max_answer:
                return False
            return True

        for annotation_, prediction_ in zip(annotation, prediction):
            start_indexes = _get_best_indexes(prediction_.start_logits, self.n_best_size)
            end_indexes = _get_best_indexes(prediction_.end_logits, self.n_best_size)
            valid_start_indexes = []
            valid_end_indexes = []
            tokens = []

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if _check_indexes(start_index, end_index, len(annotation_.tokens), self.max_answer):
                        valid_start_indexes.append(start_index)
                        valid_end_indexes.append(end_index)
                        tokens.append(annotation_.tokens[start_index:(end_index + 1)])

            start_logits = prediction_.start_logits[valid_start_indexes]
            end_logits = prediction_.end_logits[valid_end_indexes]

            start_indexes = [val for _, val in sorted(zip(start_logits+end_logits, start_indexes), reverse=True)]
            if not start_indexes:
                continue
            start_indexes_ = start_indexes[0]
            end_indexes_ = [val for _, val in sorted(zip(start_logits+end_logits, end_indexes), reverse=True)]
            end_indexes_ = end_indexes_[0]

            prediction_.start_index.append(start_indexes_)
            prediction_.end_index.append(end_indexes_)

            tokens_ = [" ".join(tok) for _, tok in sorted(zip(start_logits+end_logits, tokens), reverse=True)]
            tokens_ = tokens_[0]
            tokens_ = tokens_.replace(" ##", "")
            tokens_ = tokens_.replace("##", "")
            tokens_ = tokens_.strip()
            prediction_.tokens.append(tokens_)

        return annotation, prediction
