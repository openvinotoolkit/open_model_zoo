import numpy as np
import cv2
from .postprocessor import Postprocessor
from ..representation import (
    SuperResolutionPrediction, SuperResolutionAnnotation,
    ImageProcessingAnnotation, ImageProcessingPrediction,
    StyleTransferAnnotation, StyleTransferPrediction
)


class RGB2GRAYAnnotation(Postprocessor):
    __provider__ = 'rgb_to_gray'

    annotation_types = (SuperResolutionAnnotation, ImageProcessingAnnotation, StyleTransferAnnotation)
    prediction_types = (SuperResolutionPrediction, ImageProcessingPrediction, StyleTransferPrediction)

    def process_image(self, annotation, prediction):
        for annotation_ in annotation:
            annotation_.value = np.expand_dims(cv2.cvtColor(annotation_.value, cv2.COLOR_RGB2GRAY), -1)

        return annotation, prediction


class BGR2GRAYAnnotation(Postprocessor):
    __provider__ = 'bgr_to_gray'

    annotation_types = (SuperResolutionAnnotation, ImageProcessingAnnotation, StyleTransferAnnotation)
    prediction_types = (SuperResolutionPrediction, ImageProcessingPrediction, StyleTransferPrediction)

    def process_image(self, annotation, prediction):
        for annotation_ in annotation:
            annotation_.value = np.expand_dims(cv2.cvtColor(annotation_.value, cv2.COLOR_BGR2GRAY), -1)

        return annotation, prediction
