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

import re
from collections import Counter
import string

from ..representation import QuestionAnsweringAnnotation, QuestionAnsweringPrediction
from .metric import PerImageEvaluationMetric


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

class ScoreF1(PerImageEvaluationMetric):
    __provider__ = 'f1'

    annotation_types = (QuestionAnsweringAnnotation,)
    prediction_types = (QuestionAnsweringPrediction,)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.per_question_results = {}

    def update(self, annotation, prediction):
        gold_answers = [answer["text"] for answer in annotation.orig_answer_text if normalize_answer(answer["text"])]
        if not gold_answers:
            gold_answers = ['']
        prediction_answer = prediction.tokens[0] if prediction.tokens else ''
        max_f1_score = max(self.compute_f1(a, prediction_answer) for a in gold_answers)
        current_max_f1_score = self.per_question_results.get(annotation.question_id, 0)
        self.per_question_results[annotation.question_id] = max(max_f1_score, current_max_f1_score)
        return max_f1_score

    @staticmethod
    def compute_f1(a_gold, a_pred):
        gold_toks = get_tokens(a_gold)
        pred_toks = get_tokens(a_pred)
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def evaluate(self, annotation, prediction):
        return sum(self.per_question_results.values()) / len(self.per_question_results)

    def reset(self):
        del self.per_question_results
        self.per_question_results = {}


class ExactMatchScore(PerImageEvaluationMetric):
    __provider__ = 'exact_match'

    annotation_types = (QuestionAnsweringAnnotation,)
    prediction_types = (QuestionAnsweringPrediction,)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.per_question_results = {}

    def update(self, annotation, prediction):
        gold_answers = [answer["text"] for answer in annotation.orig_answer_text if normalize_answer(answer["text"])]
        if not gold_answers:
            gold_answers = ['']
        pred_answer = prediction.tokens[0] if prediction.tokens else ''
        max_exact_match = max(self.compute_exact(a_gold, pred_answer) for a_gold in gold_answers)
        self.per_question_results[annotation.question_id] = max(
            max_exact_match, self.per_question_results.get(annotation.question_id, 0)
        )
        return max_exact_match

    @staticmethod
    def compute_exact(a_gold, a_pred):
        return int(normalize_answer(a_gold) == normalize_answer(a_pred))

    def evaluate(self, annotation, prediction):
        return sum(self.per_question_results.values()) / len(self.per_question_results)

    def reset(self):
        del self.per_question_results
        self.per_question_results = {}
