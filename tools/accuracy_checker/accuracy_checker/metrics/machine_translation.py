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

import re
from collections import Counter
import math
from .metric import PerImageEvaluationMetric
from ..config import BoolField, NumberField, StringField
from ..representation import MachineTranslationPrediction, MachineTranslationAnnotation

class TokenizerRegexp:
    def __init__(self):
        self._re = [
            # language-dependent part (assuming Western languages)
            (re.compile(r'([\{-\~\[-\` -\&\(-\+\:-\@\/])'), r' \1 '),
            # tokenize period and comma unless preceded by a digit
            (re.compile(r'([^0-9])([\.,])'), r'\1 \2 '),
            # tokenize period and comma unless followed by a digit
            (re.compile(r'([\.,])([^0-9])'), r' \1 \2'),
            # tokenize dash when preceded by a digit
            (re.compile(r'([0-9])(-)'), r'\1 \2 '),
            # one space only between words
            (re.compile(r'\s+'), r' '),
        ]

    def __call__(self, line):
        """Common post-processing tokenizer for `13a` and `zh` tokenizers.
        :param line: a segment to tokenize
        :return: the tokenized line
        """
        for (_re, repl) in self._re:
            line = _re.sub(repl, line)

        # no leading or trailing spaces
        return line.strip()


class Tokenizer:
    def __init__(self, lower_case=False):
        self.lower_case = lower_case
        self._post_tokenizer = TokenizerRegexp()

    def __call__(self, line):
        """Tokenizes an input line using a relatively minimal tokenization
        that is however equivalent to mteval-v13a, used by WMT.
        :param line: a segment to tokenize
        :return: the tokenized line
        """
        if self.lower_case:
            line = line.lower()

        # language-independent part:
        line = line.replace('<skipped>', '')
        line = line.replace('-\n', '')
        line = line.replace('\n', ' ')
        line = line.replace('&quot;', '"')
        line = line.replace('&amp;', '&')
        line = line.replace('&lt;', '<')
        line = line.replace('&gt;', '>')

        line = " {} ".format(line)
        return self._post_tokenizer(line)

SMOOTH_DEFAULTS = {
        'floor': 0.0,
        'add-k': 1,
        'exp': None,    # No value is required
        'none': None,   # No value is required
}

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
                ),
                'smooth_method': StringField(optional=True, default='exp', choices=SMOOTH_DEFAULTS),
                'smooth_value': NumberField(optional=True),
                'lower_case': BoolField(optional=True, default=False)
            }
        )

        return parameters

    def configure(self):
        self.smooth = self.get_value_from_config('smooth')
        self.max_order = self.get_value_from_config('max_order')
        self.sys_len = 0
        self.ref_len = 0
        self.correct = [0] * self.max_order
        self.total = [0] * self.max_order
        self.lower_case = self.get_value_from_config('lower_case')
        self.tokenizer = Tokenizer(self.lower_case)
        if self.smooth:
            self.smooth_method = self.get_value_from_config('smooth_method')
            self.smooth_value = self.get_value_from_config('smooth_valoe') or SMOOTH_DEFAULTS[self.smooth_method]
        else:
            self.smooth_method = 'none'
            self.smooth_value = None

    def update(self, annotation, prediction):
        reference_corpus = annotation.reference
        translation_corpus = prediction.translation
        for lines in zip([translation_corpus], [reference_corpus]):
            output, *refs = [self.tokenizer(' '.join(x)) for x in lines]

            output_len = len(output.split())
            ref_ngrams, _, closest_len = self.reference_stats(refs, output_len)

            self.sys_len += output_len
            self.ref_len += closest_len

            sys_ngrams = self.extract_ngrams(output, self.max_order)
            for ngram in sys_ngrams.keys():
                n = len(ngram.split())
                self.correct[n - 1] += min(sys_ngrams[ngram], ref_ngrams.get(ngram, 0))
                self.total[n - 1] += sys_ngrams[ngram]

    def evaluate(self, annotations, predictions):
        def log(num):
            if num == 0.0:
                return -9999999999
            return math.log(num)

        smooth_mteval = 1.
        precisions = [0] * self.max_order

        for n in range(1, self.max_order + 1):
            if self.smooth_method == 'add-k' and n > 1:
                self.correct[n - 1] += self.smooth_value
                self.total[n - 1] += self.smooth_value

            if self.total[n - 1] == 0:
                break

            if self.correct[n - 1] == 0:
                if self.smooth_method == 'exp':
                    smooth_mteval *= 2
                    precisions[n - 1] = 1 / (smooth_mteval * self.total[n - 1])
                elif self.smooth_method == 'floor':
                    precisions[n - 1] = 1. / (smooth_mteval * self.total[n - 1])
            else:
                precisions[n - 1] = self.correct[n - 1] / self.total[n - 1]

        if self.sys_len < self.ref_len:
            bp = math.exp(1 - self.ref_len / self.sys_len) if self.sys_len > 0 else 0.0
        else:
            bp = 1.0

        score = bp * math.exp(
            sum(map(log, precisions[:self.max_order])) / self.max_order)
        pred_text, ref_text = [], []
        for pred, annot in zip(predictions, annotations):
            pred_text.append(' '.join(pred.translation))
            ref_text.append(' '.join(annot.reference))
        return score

    @staticmethod
    def extract_ngrams(segment, max_order, min_order=1):
        ngrams = Counter()
        tokens = segment.split()
        for n in range(min_order, max_order + 1):
            for i in range(0, len(tokens) - n + 1):
                ngram = ' '.join(tokens[i: i + n])
                ngrams[ngram] += 1

        return ngrams

    def reference_stats(self, refs, output_len):
        ngrams = Counter()
        closest_diff = None
        closest_len = None

        for ref in refs:
            tokens = ref.split()
            reflen = len(tokens)
            diff = abs(output_len - reflen)
            if closest_diff is None or diff < closest_diff:
                closest_diff = diff
                closest_len = reflen
            elif diff == closest_diff:
                if reflen < closest_len:
                    closest_len = reflen

            ngrams_ref = self.extract_ngrams(ref, self.max_order)
            for ngram in ngrams_ref.keys():
                ngrams[ngram] = max(ngrams[ngram], ngrams_ref[ngram])

        return ngrams, closest_diff, closest_len

    def reset(self):
        self.sys_len = 0
        self.ref_len = 0
        self.correct = [0] * self.max_order
        self.total = [0] * self.max_order
