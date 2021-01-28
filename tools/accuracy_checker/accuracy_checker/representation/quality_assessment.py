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
from .base_representation import BaseRepresentation

class QualityAssessment(BaseRepresentation):
    pass

class QualityAssessmentAnnotation(QualityAssessment):
    def __init__(self, identifier, quality_score=None):
        super().__init__(identifier)
        self.quality_score = quality_score

class QualityAssessmentPrediction(QualityAssessment):
    def __init__(self, identifier, quality_assessment, quality_score=0):
        super().__init__(identifier)
        self.quality_assessment = quality_assessment
        self.quality_score = quality_score
