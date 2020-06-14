import numpy as np
from ..postprocessor import Postprocessor
from ..representation import SuperResolutionPrediction, SuperResolutionAnnotation
from ..config import NumberField
from ..utils import get_size_from_config

try:
    from PIL import Image
except ImportError:
    Image = None


class ResizeSuperResolution(Postprocessor):
    __provider__ = 'resize_super_resolution'

    annotation_types = (SuperResolutionAnnotation, )
    prediction_types = (SuperResolutionPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'dst_width': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination width for resizing."
            ),
            'dst_height': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination height for resizing."
            )
        })

        return parameters

    def configure(self):
        if Image is None:
            raise ValueError('{} requires pillow, please install it'.format(self.__provider__))
        self.dst_height, self.dst_width = get_size_from_config(self.config, allow_none=True)

    def process_image(self, annotation, prediction):
        for annotation_, prediction_ in zip(annotation, prediction):
            target_height = self.dst_height or annotation_.shape[0]
            target_width = self.dst_width or annotation_.shape[1]
            data = Image.fromarray(prediction_.value)
            data = data.resize((target_width, target_height), Image.BICUBIC)
            prediction_.value = np.array(data)

        return annotation, prediction
