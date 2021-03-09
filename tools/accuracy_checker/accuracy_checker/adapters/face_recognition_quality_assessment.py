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

from .adapter import Adapter
from ..representation import QualityAssessmentPrediction

class QualityAssessmentAdapter(Adapter):
    __provider__ = 'face_recognition_quality_assessment'
    predcition_types = (QualityAssessmentPrediction, )
    score_weight = [0.25, 0.50, 0.50, 0.75]

    def process(self, raw, identifiers=None, frame_meta=None):
        prediction = self._extract_predictions(raw, frame_meta)[self.output_blob]
        return [QualityAssessmentPrediction(identifier, quality_assessment=self.calculate_quality(embedding))
                for identifier, embedding in zip(identifiers, prediction)]

    def calculate_quality(self, qa_feature):
        # Sort by feature value and get top1 and top2 id
        softed_feature_idx = np.argsort(qa_feature)[::-1]

        top_id1 = softed_feature_idx[0]
        top_score1 = qa_feature[top_id1]

        top_id2 = softed_feature_idx[1]
        top_score2 = qa_feature[top_id2]

        # Calculate quality score
        qs_val = 0.0
        if top_score1 > (top_score2 * 2):
            if top_id1 <= 1:
                qs_val = self.score_weight[top_id1] - 0.25 * top_score1
            else:
                qs_val = self.score_weight[top_id1] + 0.25 * top_score1
        else:
            score_sum = top_score1 + top_score2

            # Get top1 val
            top1_w = top_score1 / score_sum
            max_val1 = 0.0
            qs_val1 = 0.0

            if top_id1 <= 1:
                max_val1 = self.score_weight[top_id1]
                qs_val1 = max_val1 - 0.25 * top1_w
            else:
                qs_val1 = self.score_weight[top_id1] + 0.25 * top1_w
                max_val1 = self.score_weight[top_id1] + 0.25

            # Get top2 val
            top2_w = top_score2 / score_sum
            max_val2 = 0.0
            qs_val2 = 0.0

            if top_id2 <= 1:
                max_val2 = self.score_weight[top_id2]
                qs_val2 = max_val2 - 0.25 * top2_w
            else:
                qs_val2 = self.score_weight[top_id2] + 0.25 * top2_w
                max_val2 = self.score_weight[top_id2] + 0.25

            qs_val = (qs_val1 + qs_val2) / (max_val1 + max_val2)

        return round(qs_val, 6)
