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

from .metric_executor import MetricsExecutor
from .metric import Metric, PerImageMetricResult

from .classification import (
    ClassificationAccuracy,
    ClassificationAccuracyClasses,
    ClipAccuracy,
    ClassificationF1Score,
    MetthewsCorrelation
)
from .detection import (DetectionMAP, MissRate, Recall, DetectionAccuracyMetric, YoutubeFacesAccuracy)
from .reid import CMCScore, ReidMAP, PairwiseAccuracy, PairwiseAccuracySubsets, FaceRecognitionTAFAPairMetric
from .semantic_segmentation import SegmentationAccuracy, SegmentationIOU, SegmentationMeanAccuracy, SegmentationFWAcc
from .character_recognition import CharacterRecognitionAccuracy, LabelLevelRecognitionAccuracy
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
    StructuralSimilarity,

    AngleError
)
from .multilabel_recognition import MultiLabelRecall, MultiLabelPrecision, MultiLabelAccuracy, F1Score
from .text_detection import (
    FocusedTextLocalizationPrecision,
    FocusedTextLocalizationRecall,
    FocusedTextLocalizationHMean,
    IncidentalSceneTextLocalizationPrecision,
    IncidentalSceneTextLocalizationRecall,
    IncidentalSceneTextLocalizationHMean
)
from .coco_metrics import MSCOCOAveragePrecision, MSCOCORecall, MSCOCOKeypointsPrecision, MSCOCOKeypointsRecall
from .coco_orig_metrics import (
    MSCOCOorigAveragePrecision,
    MSCOCOorigRecall,

    MSCOCOOrigSegmAveragePrecision,
    MSCOCOorigSegmRecall,

    MSCOCOOrigKeyPointsAveragePrecision,
)
from .hit_ratio import HitRatioMetric, NDSGMetric
from .machine_translation import BilingualEvaluationUnderstudy
from .question_answering import ExactMatchScore, ScoreF1
from .mpjpe_multiperson import MpjpeMultiperson
from .language_modeling import ScorePerplexity

__all__ = [
    'Metric',
    'MetricsExecutor',
    'PerImageMetricResult',

    'ClassificationAccuracy',
    'ClassificationAccuracyClasses',
    'ClipAccuracy',
    'ClassificationF1Score',

    'DetectionMAP',
    'MissRate',
    'Recall',
    'DetectionAccuracyMetric',
    'YoutubeFacesAccuracy',

    'CMCScore',
    'ReidMAP',
    'PairwiseAccuracy',
    'PairwiseAccuracySubsets',
    'FaceRecognitionTAFAPairMetric',

    'SegmentationAccuracy',
    'SegmentationIOU',
    'SegmentationMeanAccuracy',
    'SegmentationFWAcc',

    'CharacterRecognitionAccuracy',
    'LabelLevelRecognitionAccuracy',

    'MeanAbsoluteError',
    'MeanSquaredError',
    'MeanAbsoluteErrorOnInterval',
    'MeanSquaredErrorOnInterval',
    'RootMeanSquaredError',
    'RootMeanSquaredErrorOnInterval',
    'FacialLandmarksPerPointNormedError',
    'FacialLandmarksNormedError',
    'PeakSignalToNoiseRatio',
    'StructuralSimilarity',
    'AngleError',

    'MultiLabelAccuracy',
    'MultiLabelRecall',
    'MultiLabelPrecision',
    'F1Score',

    'FocusedTextLocalizationHMean',
    'FocusedTextLocalizationRecall',
    'FocusedTextLocalizationPrecision',
    'IncidentalSceneTextLocalizationPrecision',
    'IncidentalSceneTextLocalizationRecall',
    'IncidentalSceneTextLocalizationHMean',

    'MSCOCOAveragePrecision',
    'MSCOCORecall',
    'MSCOCOKeypointsPrecision',
    'MSCOCOKeypointsRecall',
    'MSCOCOorigAveragePrecision',
    'MSCOCOorigRecall',
    'MSCOCOOrigSegmAveragePrecision',
    'MSCOCOorigSegmRecall',
    'MSCOCOOrigKeyPointsAveragePrecision',

    'HitRatioMetric',
    'NDSGMetric',

    'BilingualEvaluationUnderstudy',

    'ScoreF1',
    'ExactMatchScore',

    'MpjpeMultiperson',

    'ScorePerplexity',
]
