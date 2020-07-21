import numpy as np
import cv2
from .postprocessor import Postprocessor
from ..representation import SuperResolutionPrediction, SuperResolutionAnnotation


class RGB2GRAYSuperResolution(Postprocessor):
    __provider__ = 'rgb_to_gray_super_resolution'

    annotation_types = (SuperResolutionAnnotation, )
    prediction_types = (SuperResolutionPrediction, )

    def process_image(self, annotation, prediction):
        for annotation_ in annotation:
            annotation_.value = np.expand_dims(cv2.cvtColor(annotation_.value, cv2.COLOR_RGB2GRAY), -1)

        return annotation, prediction
