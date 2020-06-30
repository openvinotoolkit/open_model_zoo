import numpy as np
from ..postprocessor import Postprocessor
from ..representation import StyleTransferAnnotation, StyleTransferPrediction
from ..config import NumberField
from ..utils import get_size_from_config

try:
    from PIL import Image
except ImportError:
    Image = None


class ResizeStyleTransfer(Postprocessor):
    __provider__ = 'resize_style_transfer'

    annotation_types = (StyleTransferAnnotation, )
    prediction_types = (StyleTransferPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'dst_width': NumberField(
                value_type=int, optional=False, min_value=1, description="Destination width for resizing."
            ),
            'dst_height': NumberField(
                value_type=int, optional=False, min_value=1, description="Destination height for resizing."
            )
        })
        return parameters

    def configure(self):
        if Image is None:
            raise ValueError('{} requires pillow, please install it'.format(self.__provider__))
        self.dst_height, self.dst_width = get_size_from_config(self.config, allow_none=True)

    def process_image(self, annotation, prediction):
        for target in annotation:
            data = Image.fromarray(target.value)
            data = data.resize((self.dst_width, self.dst_height), Image.BICUBIC)
            target.value = np.array(data)
        return annotation, prediction
