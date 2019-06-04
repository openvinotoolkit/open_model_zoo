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

from .metric_executor import MetricsExecutor, Metric

from .classification import ClassificationAccuracy, ClassificationAccuracyClasses, ClipAccuracy
from .detection import (DetectionMAP, MissRate, Recall, DetectionAccuracyMetric)
from .reid import CMCScore, ReidMAP, PairwiseAccuracy, PairwiseAccuracySubsets
from .semantic_segmentation import SegmentationAccuracy, SegmentationIOU, SegmentationMeanAccuracy, SegmentationFWAcc
from .character_recognition import CharacterRecognitionAccuracy
from .regression import (
    MeanAbsoluteErrorOnInterval,
    MeanSquaredErrorOnInterval,

    MeanAbsoluteError,
    MeanSquaredError,

    RootMeanSquaredErrorOnInterval,
    RootMeanSquaredError,

    FacialLandmarksPerPointNormedError,
    FacialLandmarksNormedError,

    PeakSignalToNoiseRatio,

    AngleError
)
from .multilabel_recognition import MultiLabelRecall, MultiLabelPrecision, MultiLabelAccuracy, F1Score
from .text_detection import TextDetectionMetric
from .coco_metrics import MSCOCOAveragePresicion
from .coco_orig_metrics import (
    MSCOCOorigAveragePrecision,
    MSCOCOorigRecall,

    MSCOCOOrigSegmAveragePrecision,
    MSCOCOorigSegmRecall
)
from .hit_ratio import HitRatioMetric, NDSGMetric


__all__ = [
    'Metric',
    'MetricsExecutor',

    'ClassificationAccuracy',
    'ClassificationAccuracyClasses',
    'ClipAccuracy',

    'DetectionMAP',
    'MissRate',
    'Recall',
    'DetectionAccuracyMetric',

    'CMCScore',
    'ReidMAP',
    'PairwiseAccuracy',
    'PairwiseAccuracySubsets',

    'SegmentationAccuracy',
    'SegmentationIOU',
    'SegmentationMeanAccuracy',
    'SegmentationFWAcc',

    'CharacterRecognitionAccuracy',

    'MeanAbsoluteError',
    'MeanSquaredError',
    'MeanAbsoluteErrorOnInterval',
    'MeanSquaredErrorOnInterval',
    'RootMeanSquaredError',
    'RootMeanSquaredErrorOnInterval',
    'FacialLandmarksPerPointNormedError',
    'FacialLandmarksNormedError',
    'PeakSignalToNoiseRatio',
    'AngleError',

    'MultiLabelAccuracy',
    'MultiLabelRecall',
    'MultiLabelPrecision',
    'F1Score',

    'TextDetectionMetric',

    'MSCOCOAveragePresicion',
    'MSCOCOorigAveragePrecision',
    'MSCOCOorigRecall',
    'MSCOCOOrigSegmAveragePrecision',
    'MSCOCOorigSegmRecall',

    'HitRatioMetric',
    'NDSGMetric'
]
