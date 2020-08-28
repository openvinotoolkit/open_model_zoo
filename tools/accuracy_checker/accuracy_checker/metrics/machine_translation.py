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

from collections import Counter
import math
from .metric import PerImageEvaluationMetric
from ..config import BoolField, NumberField
from ..representation import MachineTranslationPrediction, MachineTranslationAnnotation

def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.
    Args:
        segment: text segment from which n-grams will be extracted.
        max_order: maximum length in tokens of the n-grams returned by this
        methods.
    Returns:
        The Counter containing all n-grams upto max_order in segment
        with a count of how many times each n-gram occurred.
    """
    ngram_counts = Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i+order])
            ngram_counts[ngram] += 1

    return ngram_counts


class BilingualEvaluationUnderstudy(PerImageEvaluationMetric):
    __provider__ = 'bleu'
    annotation_types = (MachineTranslationAnnotation, )
    prediction_types = (MachineTranslationPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update(
            {
                'smooth': BoolField(
                    optional=True, description='Whether or not to apply Lin et al. 2004 smoothing.', default=False
                ),
                'max_order': NumberField(
                    value_type=int, optional=True, description='Maximum n-gram order to use when computing BLEU score.',
                    default=4
                )
            }
        )

        return parameters

    def configure(self):
        self.smooth = self.get_value_from_config('smooth')
        self.max_order = self.get_value_from_config('max_order')
        self.matches_by_order = [0] * self.max_order
        self.possible_matches_by_order = [0] * self.max_order
        self.reference_length = 0
        self.translation_length = 0

    def update(self, annotation, prediction):
        reference_corpus = annotation.reference
        translation_corpus = prediction.translation
        for (references, translation) in zip(reference_corpus, translation_corpus):
            self.reference_length += min(len(r) for r in references)
            self.translation_length += len(translation)

            merged_ref_ngram_counts = self._get_ngrams(references, self.max_order)
            translation_ngram_counts = self._get_ngrams(translation, self.max_order)
            overlap = translation_ngram_counts & merged_ref_ngram_counts
            for ngram in overlap:
                self.matches_by_order[len(ngram) - 1] += overlap[ngram]
            for order in range(1, self.max_order + 1):
                possible_matches = len(translation) - order + 1
                if possible_matches > 0:
                    self.possible_matches_by_order[order - 1] += possible_matches

    def evaluate(self, annotations, predictions):
        precisions = [0] * self.max_order
        for i in range(0, self.max_order):
            if self.smooth:
                precisions[i] = ((self.matches_by_order[i] + 1.) / (self.possible_matches_by_order[i] + 1.))
            else:
                if self.possible_matches_by_order[i] > 0:
                    precisions[i] = (float(self.matches_by_order[i]) / self.possible_matches_by_order[i])
                else:
                    precisions[i] = 0.0

        if min(precisions) > 0:
            p_log_sum = sum((1. / self.max_order) * math.log(p) for p in precisions)
            geo_mean = math.exp(p_log_sum)
        else:
            geo_mean = 0

        ratio = float(self.translation_length) / self.reference_length
        if ratio > 1.0:
            bp = 1.
        else:
            bp = math.exp(1 - 1. / ratio)

        bleu = geo_mean * bp

        return bleu

    @staticmethod
    def _get_ngrams(segment, max_order):
        """Extracts all n-grams upto a given maximum order from an input segment.
        Args:
          segment: text segment from which n-grams will be extracted.
          max_order: maximum length in tokens of the n-grams returned by this
              methods.
        Returns:
          The Counter containing all n-grams upto max_order in segment
          with a count of how many times each n-gram occurred.
        """
        ngram_counts = Counter()
        for order in range(1, max_order + 1):
            for i in range(0, len(segment) - order + 1):
                ngram = tuple(segment[i:i + order])
                ngram_counts[ngram] += 1

        return ngram_counts
