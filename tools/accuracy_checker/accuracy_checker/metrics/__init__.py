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

from .metric_executor import MetricsExecutor
from .metric import Metric, PerImageMetricResult

from .classification import (
    ClassificationAccuracy,
    ClassificationAccuracyClasses,
    ClipAccuracy,
    ClassificationF1Score,
    MetthewsCorrelation,
    RocAucScore,
    AcerScore,
)
from .detection import (
    DetectionMAP,
    MissRate,
    Recall,
    DetectionAccuracyMetric,
    YoutubeFacesAccuracy
)
from .reid import (
    CMCScore,
    ReidMAP,
    PairwiseAccuracy,
    PairwiseAccuracySubsets,
    FaceRecognitionTAFAPairMetric,
    NormalizedEmbeddingAccuracy,
    LocalizationRecall
)
from .semantic_segmentation import SegmentationAccuracy, SegmentationIOU, SegmentationMeanAccuracy, SegmentationFWAcc
from .character_recognition import CharacterRecognitionAccuracy, LabelLevelRecognitionAccuracy
from .regression import (
    MeanAbsoluteErrorOnInterval,
    MeanSquaredErrorOnInterval,

    MeanAbsoluteError,
    MeanSquaredError,
    Log10Error,
    MeanAbsolutePercentageError,
    RelativeL2Error,

    RootMeanSquaredErrorOnInterval,
    RootMeanSquaredError,

    FacialLandmarksPerPointNormedError,
    FacialLandmarksNormedError,

    PeakSignalToNoiseRatio,
    StructuralSimilarity,

    AngleError,

    PercentageCorrectKeypoints
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
from .question_answering import ExactMatchScore, ScoreF1, QuestionAnsweringEmbeddingAccuracy
from .ner import NERAccuracy, NERFScore, NERPrecision, NERRecall
from .mpjpe_multiperson import MpjpeMultiperson
from .language_modeling import ScorePerplexity

from .attribute_classification import (
    AttributeClassificationRecall,
    AttributeClassificationPrecision,
    AttributeClassificationAccuracy
)
from .im2latex_images_match import Im2latexRenderBasedMetric

from .speech_recognition import SpeechRecognitionWER, SpeechRecognitionCER
from .score_class_comparison import ScoreClassComparisonMetric
from .dna_seq_accuracy import DNASequenceAccuracy

from .gan_metrics import InceptionScore, FrechetInceptionDistance

from .salient_objects_detection import SalienceMapMAE, SalienceEMeasure, SalienceMapFMeasure, SalienceSMeasure

__all__ = [
    'Metric',
    'MetricsExecutor',
    'PerImageMetricResult',

    'ClassificationAccuracy',
    'ClassificationAccuracyClasses',
    'ClipAccuracy',
    'ClassificationF1Score',
    'MetthewsCorrelation',

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
    'NormalizedEmbeddingAccuracy',
    'LocalizationRecall',

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
    'MeanAbsolutePercentageError',
    'Log10Error',
    'RelativeL2Error',

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
    'QuestionAnsweringEmbeddingAccuracy',

    'NERAccuracy',
    'NERPrecision',
    'NERRecall',
    'NERFScore',

    'MpjpeMultiperson',

    'ScorePerplexity',

    'AttributeClassificationRecall',
    'AttributeClassificationPrecision',
    'AttributeClassificationAccuracy',

    'SpeechRecognitionWER',
    'SpeechRecognitionCER',

    'ScoreClassComparisonMetric',

    'RocAucScore',

    'Im2latexRenderBasedMetric',

    'PercentageCorrectKeypoints',

    'DNASequenceAccuracy',

    'InceptionScore',
    'FrechetInceptionDistance',

    'AcerScore',

    'SalienceMapMAE',
    'SalienceMapFMeasure',
    'SalienceSMeasure',
    'SalienceEMeasure'
]
