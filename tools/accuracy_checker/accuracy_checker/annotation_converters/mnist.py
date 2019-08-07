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
from ..utils import read_csv, check_file_existence

from .format_converter import BaseFormatConverter, ConverterReturn


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

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        """
        This method is executed automatically when convert.py is started.
        All arguments are automatically got from command line arguments or config file in method configure

        Returns:
            annotations: list of annotation representation objects.
            meta: dictionary with additional dataset level metadata.
        """
        annotations = []
        check_images = check_content and not self.convert_images
        content_errors = None
        if check_content:
            self.converted_images_dir = self.converted_images_dir or self.test_csv_file.parent / 'converted_images'

        if self.converted_images_dir and check_content:
            if not self.converted_images_dir.exists():
                content_errors = ['{}: does not exist'.format(self.converted_images_dir)]
                check_images = False
        # read original dataset annotation
        annotation_table = read_csv(self.test_csv_file)
        num_iterations = len(annotation_table)
        for index, annotation in enumerate(annotation_table):
            identifier = '{}.png'.format(index)
            label = int(annotation['label'])
            if self.convert_images:
                image = Image.fromarray(self.convert_image(annotation))
                image = image.convert("L")
                image.save(str(self.converted_images_dir / identifier))
            annotations.append(ClassificationAnnotation(identifier, label))
            if check_images:
                if not check_file_existence(self.converted_images_dir / identifier):
                    # add error to errors list if file not found
                    content_errors.append('{}: does not exist'.format(self.converted_images_dir / identifier))

            if progress_callback is not None and index % progress_interval == 0:
                progress_callback(index / num_iterations * 100)

        meta = {'label_map': {str(i): i for i in range(10)}}

        return ConverterReturn(annotations, meta, None)

    @staticmethod
    def convert_image(features):
        image = np.zeros((28, 28))
        column_template = '{}x{}'
        for x in range(28):
            for y in range(28):
                pixel = int(features[column_template.format(x+1, y+1)])
                image[x, y] = pixel

        return image
