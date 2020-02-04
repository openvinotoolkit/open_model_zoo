from PIL import Image
import numpy as np
from ..postprocessor import Postprocessor
from ..representation import SuperResolutionPrediction, SuperResolutionAnnotation


class ResizeSuperResolution(Postprocessor):
    __provider__ = 'resize_super_resolution'

    annotation_types = (SuperResolutionAnnotation, )
    prediction_types = (SuperResolutionPrediction, )

    def process_image(self, annotation, prediction):
        for annotation_, prediction_ in zip(annotation, prediction):
            h, w, _ = annotation_.value.shape
            data = Image.fromarray(prediction_.value)
            data = data.resize((w, h), Image.BICUBIC)
            prediction_.value = np.array(data)
        return annotation, prediction
