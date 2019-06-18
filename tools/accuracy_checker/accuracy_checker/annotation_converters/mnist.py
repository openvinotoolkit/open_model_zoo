"""
Copyright (c) 2019 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from PIL import Image
import numpy as np
from ..config import PathField, BoolField
from ..representation import ClassificationAnnotation
from ..utils import read_csv

from .format_converter import BaseFormatConverter


class MNISTCSVFormatConverter(BaseFormatConverter):
    """
    MNIST CSV dataset converter. All annotation converters should be derived from BaseFormatConverter class.
    """

    # register name for this converter
    # this name will be used for converter class look up
    __provider__ = 'mnist_csv'
    annotation_types = (ClassificationAnnotation, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'annotation_file': PathField(description="Path to csv file which contain dataset."),
            'convert_images': BoolField(
                optional=True,
                default=False,
                description="Allows to convert images from pickle file to user specified directory."
            ),
            'converted_images_dir': PathField(
                optional=True, is_directory=True, check_exists=False, description="Path to converted images location."
            ),
        })

        return parameters

    def configure(self):
        """
        This method is responsible for obtaining the necessary parameters
        for converting from the command line or config.
        """
        self.test_csv_file = self.get_value_from_config('annotation_file')
        self.converted_images_dir = self.get_value_from_config('converted_images_dir')
        self.convert_images = self.get_value_from_config('convert_images')
        if self.convert_images and not self.converted_images_dir:
            self.converted_images_dir = self.test_csv_file.parent / 'converted_images'
            if not self.converted_images_dir.exists():
                self.converted_images_dir.mkdir(parents=True)

    def convert(self):
        """
        This method is executed automatically when convert.py is started.
        All arguments are automatically got from command line arguments or config file in method configure

        Returns:
            annotations: list of annotation representation objects.
            meta: dictionary with additional dataset level metadata.
        """
        annotations = []
        # read original dataset annotation
        annotation_table = read_csv(self.test_csv_file)
        for index, annotation in enumerate(annotation_table):
            identifier = '{}.png'.format(index)
            label = int(annotation['label'])
            if self.convert_images:
                image = Image.fromarray(self.convert_image(annotation))
                image = image.convert("L")
                image.save(str(self.converted_images_dir / identifier))
            annotations.append(ClassificationAnnotation(identifier, label))

        meta = {'label_map': {str(i): i for i in range(10)}}

        return annotations, meta

    @staticmethod
    def convert_image(features):
        image = np.zeros((28, 28))
        column_template = '{}x{}'
        for x in range(28):
            for y in range(28):
                pixel = int(features[column_template.format(x+1, y+1)])
                image[x, y] = pixel

        return image
