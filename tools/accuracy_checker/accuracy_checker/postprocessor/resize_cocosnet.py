import numpy as np
from .postprocessor import Postprocessor, PostprocessorWithSpecificTargets
from ..representation import ImageProcessingAnnotation, ImageProcessingPrediction

try:
    from PIL import Image
except ImportError:
    Image = None


class ResizeCocosnet(PostprocessorWithSpecificTargets):
    __provider__ = 'resize_cocosnet'

    annotation_types = (ImageProcessingAnnotation, )
    prediction_types = (ImageProcessingPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        return parameters

    def configure(self):
        if Image is None:
            raise ValueError('{} requires pillow, please install it'.format(self.__provider__))

    def process_image(self, annotation, prediction):
        self.target_height = self.image_size[0]
        self.target_width = self.image_size[1]
        for target in annotation:
            target.value = self.resize(target.value)

        for target in prediction:
            target.value = self.resize(target.value)

        return annotation, prediction

    def resize(self, data):
        data = Image.fromarray(data)
        data = data.resize((self.target_width, self.target_height), Image.LINEAR)
        return np.array(data)