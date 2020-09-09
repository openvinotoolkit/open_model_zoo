import numpy as np
from .postprocessor import Postprocessor, PostprocessorWithSpecificTargets
from ..representation import CocosnetAnnotation, CocosnetPrediction

try:
    from PIL import Image
except ImportError:
    Image = None


class ResizeCocosnet(PostprocessorWithSpecificTargets):
    __provider__ = 'resize_cocosnet'

    annotation_types = (CocosnetAnnotation, )
    prediction_types = (CocosnetPrediction, )

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
            target.value = self.resize(np.uint8(target.value))

        for target in prediction:
            target.value = self.resize(np.uint8(target.value * 127.5 + 127.5))

        return annotation, prediction

    def resize(self, data):
        data = Image.fromarray(data)
        data = data.resize((self.target_width, self.target_height), Image.LINEAR)
        return np.array(data)