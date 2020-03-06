import cv2
from .postprocessor import Postprocessor
from ..representation import DepthEstimationAnnotation, DepthEstimationPrediction


class ResizeDepthMap(Postprocessor):
    __provider__ = 'resize_prediction_depth_map'

    annotation_types = (DepthEstimationAnnotation, )
    prediction_types = (DepthEstimationPrediction, )

    def process_image(self, annotation, prediction):
        h, w, _ = self.image_size

        for target_prediction in prediction:
            target_prediction.depth_map = cv2.resize(target_prediction.depth_map, (w, h))

        return annotation, prediction
