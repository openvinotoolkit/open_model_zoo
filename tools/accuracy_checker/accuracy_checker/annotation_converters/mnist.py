"""
Copyright (c) 2018-2021 Intel Corporation

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

import gzip
import numpy as np
from PIL import Image
from ..config import PathField, BoolField
from ..representation import ClassificationAnnotation
from ..utils import read_csv, check_file_existence, read_json

from .format_converter import BaseFormatConverter, ConverterReturn


class MNISTFormatConverter(BaseFormatConverter):
    __provider__ = 'mnist'
    annotation_types = (ClassificationAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'labels_file': PathField(description="Path to labels in binary format."),
            'images_file': PathField(description="Path to images in binary format."),
            'convert_images': BoolField(
                optional=True,
                default=False,
                description="Allows to convert images to user specified directory."
            ),
            'converted_images_dir': PathField(
                optional=True, is_directory=True, check_exists=False, description="Path to converted images location."
            ),
            'dataset_meta_file': PathField(
                description='path to json file with dataset meta (e.g. label_map, color_encoding)', optional=True
            )
        })

        return configuration_parameters

    def configure(self):
        """
        This method is responsible for obtaining the necessary parameters
        for converting from the command line or config.
        """
        self.test_anno_file = self.get_value_from_config('labels_file')
        self.test_data_file = self.get_value_from_config('images_file')
        self.converted_images_dir = self.get_value_from_config('converted_images_dir')
        self.convert_images = self.get_value_from_config('convert_images')
        if self.convert_images and not self.converted_images_dir:
            self.converted_images_dir = self.test_anno_file.parent / 'converted_images'
        if self.convert_images and not self.converted_images_dir.exists():
            self.converted_images_dir.mkdir(parents=True)
        self.dataset_meta = self.get_value_from_config('dataset_meta_file')

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
        meta = self.generate_meta()
        content_errors = None
        if check_content:
            self.converted_images_dir = self.converted_images_dir or self.test_anno_file.parent / 'converted_images'

        if self.converted_images_dir and check_content:
            if not self.converted_images_dir.exists():
                content_errors = ['{}: does not exist'.format(self.converted_images_dir)]
                check_images = False
        # read original dataset annotation

        with gzip.open(str(self.test_anno_file), 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(str(self.test_data_file), 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

        num_iterations = len(labels)
        for index, annotation in enumerate(labels):
            identifier = '{}.png'.format(index)
            label = int(annotation)
            if self.convert_images:
                image = Image.fromarray(images[index].reshape(28, 28))
                image = image.convert("L")
                image.save(str(self.converted_images_dir / identifier))
            annotations.append(ClassificationAnnotation(identifier, label))
            if check_images:
                if not check_file_existence(self.converted_images_dir / identifier):
                    # add error to errors list if file not found
                    content_errors.append('{}: does not exist'.format(self.converted_images_dir / identifier))

            if progress_callback is not None and index % progress_interval == 0:
                progress_callback(index / num_iterations * 100)

        return ConverterReturn(annotations, meta, content_errors)

    def generate_meta(self):
        if not self.dataset_meta:
            return {'label_map': {str(i): i for i in range(10)}}
        dataset_meta = read_json(self.dataset_meta)
        label_map = dataset_meta.get('label_map')
        if 'labels' in dataset_meta:
            label_map = dict(enumerate(dataset_meta['labels']))
        dataset_meta['label_map'] = label_map or {str(i): i for i in range(10)}

        return dataset_meta


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
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'annotation_file': PathField(description="Path to csv file which contain dataset."),
            'convert_images': BoolField(
                optional=True,
                default=False,
                description="Allows to convert images from pickle file to user specified directory."
            ),
            'converted_images_dir': PathField(
                optional=True, is_directory=True, check_exists=False, description="Path to converted images location."
            ),
            'dataset_meta_file': PathField(
                description='path to json file with dataset meta (e.g. label_map, color_encoding)', optional=True
            )
        })

        return configuration_parameters

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
        if self.convert_images and not self.converted_images_dir.exists():
            self.converted_images_dir.mkdir(parents=True)

        self.dataset_meta = self.get_value_from_config('dataset_meta_file')

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
        meta = self.generate_meta()
        labels_to_id = meta['label_map']
        content_errors = None
        if check_content:
            self.converted_images_dir = self.converted_images_dir or self.test_csv_file.parent / 'converted_images'

        if self.converted_images_dir and check_content:
            if not self.converted_images_dir.exists():
                content_errors = ['{}: does not exist'.format(self.converted_images_dir)]
                check_images = False
        # read original dataset annotation
        annotation_table = read_csv(self.test_csv_file, is_dict=False)
        if annotation_table[0][0] == 'label':
            # skip header
            annotation_table = annotation_table[1:]
        num_iterations = len(annotation_table)
        for index, annotation in enumerate(annotation_table):
            identifier = '{}.png'.format(index)
            label = labels_to_id.get(annotation[0], int(annotation[0]))
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

        return ConverterReturn(annotations, meta, content_errors)

    @staticmethod
    def convert_image(features):
        image = np.array(features[1:], dtype=np.uint8).reshape((28, 28))

        return image

    def generate_meta(self):
        if not self.dataset_meta:
            return {'label_map': {str(i): i for i in range(10)}}
        dataset_meta = read_json(self.dataset_meta)
        label_map = dataset_meta.get('label_map')
        if 'labels' in dataset_meta:
            label_map = dict(enumerate(dataset_meta['labels']))
        dataset_meta['label_map'] = label_map or {str(i): i for i in range(10)}

        return dataset_meta
