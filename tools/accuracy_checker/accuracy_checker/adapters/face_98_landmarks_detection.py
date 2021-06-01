import cv2
import numpy as np

from ..adapters import Adapter
from ..config import ConfigValidator, StringField, ConfigError
from ..preprocessor import ObjectCropWithScale
from ..representation import Face98LandmarksPrediction
from ..utils import contains_any


class FaceLandmarksAdapter(Adapter):
    __provider__ = 'face_landmarks_detection'
    
    def process(self, raw, identifiers, frame_meta):  
        result = self._extract_predictions(raw, frame_meta)
        res = [Face98LandmarksPrediction(identifiers, None, None, result['3851'])]

        return res
