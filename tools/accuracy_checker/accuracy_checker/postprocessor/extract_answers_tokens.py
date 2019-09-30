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
            index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
            best_indexes_mask = [np.arange(len(index_and_score)) < n_best_size]
            best_indexes = np.array(index_and_score, dtype=np.int32)[tuple(best_indexes_mask)][..., 0]
            return best_indexes

        def _check_indexes(start, end, length, max_answer):
            if start >= length or end >= length:
                return False
            if end < start or end - start + 1 > max_answer:
                return False
            return True

        for annotation_, prediction_ in zip(annotation, prediction):
            start_indexes = [_get_best_indexes(logits, self.n_best_size) for logits in prediction_.start_logits]
            end_indexes = [_get_best_indexes(logits, self.n_best_size) for logits in prediction_.end_logits]
            for i, _ in enumerate(start_indexes):
                start_indexes_ = []
                end_indexes_ = []
                tokens_ = []

                for start_index_ in start_indexes[i]:
                    for end_index_ in end_indexes[i]:
                        if _check_indexes(start_index_, end_index_, len(annotation_.tokens), self.max_answer):
                            start_indexes_.append(start_index_)
                            end_indexes_.append(end_index_)
                            tokens_.append(annotation_.tokens[start_index_:(end_index_ + 1)])

                start_logits_ = prediction_.start_logits[i][start_indexes_]
                end_logits_ = prediction_.end_logits[i][end_indexes_]

                start_indexes_ = [val for _, val in
                                  sorted(zip(start_logits_+end_logits_, start_indexes_), reverse=True)]
                if not start_indexes_:
                    continue
                start_indexes_ = start_indexes_[0]
                end_indexes_ = [val for _, val in sorted(zip(start_logits_+end_logits_, end_indexes_), reverse=True)]
                end_indexes_ = end_indexes_[0]

                prediction_.start_index.append(start_indexes_)
                prediction_.end_index.append(end_indexes_)

                tokens_ = [" ".join(tok) for _, tok in sorted(zip(start_logits_+end_logits_, tokens_), reverse=True)]
                tokens_ = tokens_[0]
                tokens_ = tokens_.replace(" ##", "")
                tokens_ = tokens_.replace("##", "")
                tokens_ = tokens_.strip()
                prediction_.tokens.append(tokens_)

        return annotation, prediction
