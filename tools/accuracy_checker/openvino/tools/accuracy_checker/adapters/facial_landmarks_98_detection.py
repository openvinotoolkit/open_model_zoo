from ..adapters import Adapter
from ..representation import FacialLandmarksHeatMapPrediction

class FacialLandmarksAdapter(Adapter):
    __provider__ = 'facial_landmarks_detection'

    def process(self, raw, identifiers, frame_meta):
        result = self._extract_predictions(raw, frame_meta)
        res = [FacialLandmarksHeatMapPrediction(identifiers[0], None, None, result[self.output_blob])]

        return res
