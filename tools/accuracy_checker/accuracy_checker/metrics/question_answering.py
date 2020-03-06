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

import re
from collections import Counter

from ..representation import QuestionAnsweringAnnotation, QuestionAnsweringPrediction
from .metric import PerImageEvaluationMetric


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


class ScoreF1(PerImageEvaluationMetric):
    __provider__ = 'f1'

    annotation_types = (QuestionAnsweringAnnotation,)
    prediction_types = (QuestionAnsweringPrediction,)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f1 = 0
        self.total = 0

    def update(self, annotation, prediction):
        max_f1_score = 0
        for gt_answer in annotation.orig_answer_text:
            for pred_answer in prediction.tokens:
                prediction_tokens = normalize_answer(pred_answer).split()
                annotation_tokens = normalize_answer(gt_answer['text']).split()
                common = Counter(prediction_tokens) & Counter(annotation_tokens)
                same = sum(common.values())
                if same == 0:
                    continue
                precision = 1.0 * same / len(prediction_tokens)
                recall = 1.0 * same / len(annotation_tokens)
                f1 = (2 * precision * recall) / (precision + recall)
                max_f1_score = f1 if f1 > max_f1_score else max_f1_score
        self.f1 += max_f1_score
        self.total += 1
        return max_f1_score

    def evaluate(self, annotation, prediction):
        return self.f1 / self.total


class ExactMatchScore(PerImageEvaluationMetric):
    __provider__ = 'exact_match'

    annotation_types = (QuestionAnsweringAnnotation,)
    prediction_types = (QuestionAnsweringPrediction,)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exact_match = 0
        self.total = 0

    def update(self, annotation, prediction):
        max_exact_match = 0
        for gt_answer in annotation.orig_answer_text:
            for pred_answer in prediction.tokens:
                exact_match = normalize_answer(gt_answer['text']) == normalize_answer(pred_answer)
                max_exact_match = exact_match if exact_match > max_exact_match else max_exact_match
        self.exact_match += max_exact_match
        self.total += 1
        return max_exact_match

    def evaluate(self, annotation, prediction):
        return self.exact_match / self.total
