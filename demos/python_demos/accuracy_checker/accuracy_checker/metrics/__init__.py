"""
 Copyright (c) 2018 Intel Corporation

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
from .metric_executor import MetricsExecutor

from .classification import ClassificationAccuracy, ClassificationAccuracyClasses
from .detection import DetectionMAP, MissRate, Recall
from .reid import CMCScore, ReidMAP, PairwiseAccuracy, PairwiseAccuracySubsets
from .semantic_segmentation import SegmentationAccuracy, SegmentationIOU, SegmentationMeanAccuracy, SegmentationFWAcc
from .character_recognition import CharacterRecognitionAccuracy
from .regression import (MeanAbsoluteErrorOnInterval, MeanSquaredErrorOnInterval, MeanSquaredError, MeanAbsoluteError,
                         RootMeanSquaredError, RootMeanSquaredErrorOnInterval,
                         PerPointRegression, AveragePointError)
from .multilabel_recognition import MultilabelRecall, MultilabelPrecision, MultilabelAccuracy, F1Score

__all__ = [
    'MetricsExecutor', 'DetectionMAP', 'MissRate',
    'Recall', 'CMCScore', 'ReidMAP', 'ClassificationAccuracy', 'ClassificationAccuracyClasses',
    'SegmentationAccuracy', 'SegmentationIOU', 'SegmentationMeanAccuracy', 'SegmentationFWAcc',
    'CharacterRecognitionAccuracy', 'PairwiseAccuracy', 'PairwiseAccuracySubsets',
    'MeanAbsoluteError', 'MeanSquaredError', 'MeanAbsoluteErrorOnInterval', 'MeanSquaredErrorOnInterval',
    'RootMeanSquaredError', 'RootMeanSquaredErrorOnInterval', 'PerPointRegression', 'AveragePointError',
    'MultilabelAccuracy', 'MultilabelRecall', 'MultilabelPrecision', 'F1Score'
]
