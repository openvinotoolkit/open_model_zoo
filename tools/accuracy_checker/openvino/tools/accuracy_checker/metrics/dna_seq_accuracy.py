import re
from collections import defaultdict
import numpy as np
from .metric import PerImageEvaluationMetric
from ..config import BoolField, NumberField
from ..representation import (
    DNASequenceAnnotation, DNASequencePrediction, CharacterRecognitionAnnotation, CharacterRecognitionPrediction
)
from ..utils import UnsupportedPackage

try:
    import parasail
except ImportError as error:
    parasail = UnsupportedPackage('parasail', error.msg)

split_cigar = re.compile(r"(?P<len>\d+)(?P<op>\D+)")


class DNASequenceAccuracy(PerImageEvaluationMetric):
    __provider__ = 'dna_seq_accuracy'
    annotation_types = (DNASequenceAnnotation, CharacterRecognitionAnnotation)
    prediction_types = (DNASequencePrediction, CharacterRecognitionPrediction)

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'balanced': BoolField(optional=True, default=False),
            'min_coverage': NumberField(optional=True, default=0.5, min_value=0, max_value=1)
        })
        return params

    def configure(self):
        if isinstance(parasail, UnsupportedPackage):
            parasail.raise_error(self.__provider__)
        self.balanced = self.get_value_from_config('balanced')
        self.min_coverage = self.get_value_from_config('min_coverage')
        self.accuracy = []
        self.meta.update({
            'names': ['mean', 'median'],
            'calculate_mean': False
        })

    def update(self, annotation, prediction):
        alignment = parasail.sw_trace_striped_32(prediction.label, annotation.label, 8, 4, parasail.dnafull)
        counts = defaultdict(int)
        _, cigar = self._parasail_to_sam(alignment, prediction.label)

        r_coverage = len(alignment.traceback.ref) / len(annotation.label)

        if r_coverage < self.min_coverage:
            self.accuracy.append(0.0)
            return 0.0

        for count, op in re.findall(split_cigar, cigar):
            counts[op] += int(count)

        if self.balanced:
            accuracy = (counts['='] - counts['I']) / (counts['='] + counts['X'] + counts['D'])
        else:
            accuracy = counts['='] / (counts['='] + counts['I'] + counts['X'] + counts['D'])
        self.accuracy.append(accuracy)

        return accuracy

    @staticmethod
    def _parasail_to_sam(result, seq):
        cigstr = result.cigar.decode.decode()
        first = re.search(split_cigar, cigstr)

        first_count, first_op = first.groups()
        prefix = first.group()
        rstart = result.cigar.beg_ref
        cliplen = result.cigar.beg_query

        clip = '' if cliplen == 0 else '{}S'.format(cliplen)
        if first_op == 'I':
            pre = '{}S'.format(int(first_count) + cliplen)
        elif first_op == 'D':
            pre = clip
            rstart = int(first_count)
        else:
            pre = '{}{}'.format(clip, prefix)

        mid = cigstr[len(prefix):]
        end_clip = len(seq) - result.end_query - 1
        suf = '{}S'.format(end_clip) if end_clip > 0 else ''
        new_cigstr = ''.join((pre, mid, suf))

        return rstart, new_cigstr

    def evaluate(self, annotations, predictions):
        return [np.mean(self.accuracy), np.median(self.accuracy)]

    def reset(self):
        self.accuracy = []
