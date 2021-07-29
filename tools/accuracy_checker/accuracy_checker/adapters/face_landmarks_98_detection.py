from ..adapters import Adapter
from ..representation import FaceLandmarksHeatMapPrediction

class FaceLandmarksAdapter(Adapter):
    __provider__ = 'face_landmarks_detection'

    def process(self, raw, identifiers, frame_meta):
        result = self._extract_predictions(raw, frame_meta)
        res = [FaceLandmarksHeatMapPrediction(identifiers[0], None, None, result[self.output_blob])]

        return res
