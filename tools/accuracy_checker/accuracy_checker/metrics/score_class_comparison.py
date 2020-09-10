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

import numpy as np
from .metric import FullDatasetEvaluationMetric
from ..config import NumberField
from ..representation import QualityAssessmentAnnotation, QualityAssessmentPrediction

class ScoreClassComparisonMetric(FullDatasetEvaluationMetric):
    __provider__ = 'score_class_comparison'
    annotation_types = (QualityAssessmentAnnotation, )
    prediction_types = (QualityAssessmentPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'num_high_quality': NumberField(
                value_type=int,
                min_value=1,
                optional=False,
                description="The number of high-quality class in total"
            ),
            'num_low_quality': NumberField(
                value_type=int,
                min_value=1,
                optional=False,
                description="The number of low-quality class in total"
            )
        })
        return parameters

    def configure(self):
        self.high_class = self.get_value_from_config('num_high_quality')
        self.low_class = self.get_value_from_config('num_low_quality')

    def evaluate(self, annotations, predictions):
        if len(predictions) < self.high_class or len(predictions) < self.low_class:
            return 0.0

        # Sort by quality score
        quality_scores = np.array([a.quality_score for a in annotations])
        quality_scores[::-1].sort()

        th_qs_high = quality_scores[self.high_class]
        th_qs_low = quality_scores[::-1][self.low_class]

        # Sort by quality assessment
        quality_assessments = np.array([p.quality_assessment for p in predictions])
        quality_assessments[::-1].sort()

        th_qa_high = quality_assessments[self.high_class]
        th_qa_low = quality_assessments[::-1][self.low_class]

        # Set class (-1:low, 0:normal, 1:good)
        tp = 0
        tn = 0
        for annotation, prediction in zip(annotations, predictions):
            qs_class = 0
            qa_class = 0

            quality_score = annotation.quality_score
            quality_assessment = prediction.quality_assessment

            # Give class by quality score
            if quality_score > th_qs_high:
                qs_class = 1
            elif quality_score < th_qs_low:
                qs_class = -1

            # Give class by quality assessment
            if quality_assessment > th_qa_high:
                qa_class = 1
            elif quality_assessment < th_qa_low:
                qa_class = -1

            # Increase TP or TN
            if qs_class == 1 and qa_class != -1:
                tp += 1
            elif qs_class == -1 and qa_class != 1:
                tn += 1

        accuracy = (tp + tn) / (self.high_class + self.low_class)

        return accuracy
