
from PIL import Image
import numpy as np
from .format_converter import BaseFormatConverter, ConverterReturn
from ..config import PathField, StringField, BoolField
from ..representation import ClassificationAnnotation


class ClutteredMNISTConverter(BaseFormatConverter):
    __provider__ = 'cluttered_mnist'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'data_file': PathField(),
            'split': StringField(optional=True, default='test', choices=['train', 'valid', 'test']),
            'convert_images': BoolField(optional=True, default=True),
            'images_dir': PathField(is_directory=True, optional=True)

        })
        return params

    def configure(self):
        self.data_file = self.get_value_from_config('data_file')
        self.split = self.get_value_from_config('split')
        self.convert_images = self.get_value_from_config('convert_images')
        self.images_dir = self.get_value_from_config('images_dir') or self.data_file.parent / 'converted_images'
        if self.convert_images and not self.images_dir.exists():
            self.images_dir.mkdir(parents=True)

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        data = np.load(str(self.data_file))
        x_values = data['x_{}'.format(self.split)]
        y_values = data['y_{}'.format(self.split)]
        annotations = []
        for idx, y in enumerate(y_values):
            identifier = '{}_{}.png'.format(self.split, idx)
            y_label = np.argmax(y)
            if self.convert_images:
                x = x_values[idx].reshape((60, 60)) * 255
                image = Image.fromarray(x)
                image = image.convert("L")
                image.save(str(self.images_dir / identifier))
            annotations.append(ClassificationAnnotation(identifier, y_label))
        return ConverterReturn(annotations, None, None)
