from ..representation import DetectionPrediction
from .adapter import Adapter


class GVADetectionAdapter(Adapter):
    __provider__ = 'gva_detection'

    def process(self, raw, identifiers, frame_meta):
        results = []
        for identifier, prediction in zip(identifiers, raw):
            objects = prediction.get('objects', [])
            x_mins, y_mins, x_maxs, y_maxs, scores, labels = [], [], [], [], [], []
            for obj in objects:
                det = obj['detection']
                scores.append(det['confidence'])
                labels.append(det['label_id'])
                bbox = det['bounding_box']
                x_mins.append(bbox['x_min'])
                y_mins.append(bbox['y_min'])
                x_maxs.append(bbox['x_max'])
                y_maxs.append(bbox['y_max'])
            results.append(DetectionPrediction(identifier, labels, scores, x_mins, y_mins, x_maxs, y_maxs))

        return results
