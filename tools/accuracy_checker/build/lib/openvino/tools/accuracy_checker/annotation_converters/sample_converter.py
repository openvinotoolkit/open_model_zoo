"""
Copyright (c) 2018-2022 Intel Corporation

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

import re

from ..config import PathField
from ..representation import ClassificationAnnotation
from ..utils import get_path, read_txt

from .format_converter import BaseFormatConverter, ConverterReturn


class SampleConverter(BaseFormatConverter):
    """
    Sample dataset converter. All annotation converters should be derived from BaseFormatConverter class.
    """

    # register name for this converter
    # this name will be used for converter class look up
    __provider__ = 'sample'
    annotation_types = (ClassificationAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'data_dir': PathField(is_directory=True, description="Path to sample dataset root directory.")
        })

        return configuration_parameters

    def configure(self):
        """
        This method is responsible for obtaining the necessary parameters
        for converting from the command line or config.
        """
        self.data_dir = self.config['data_dir']

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        """
        This method is executed automatically when convert.py is started.
        All arguments are automatically got from command line arguments or config file in method configure

        Returns:
            annotations: list of annotation representation objects.
            meta: dictionary with additional dataset level metadata (if provided)
        """

        dataset_directory = get_path(self.data_dir, is_directory=True)
        # create dataset metadata
        metadata = self.get_meta()

        # read and convert annotation
        images_dir = dataset_directory / 'test'
        annotations = self._convert_annotations(images_dir, metadata, progress_callback, progress_interval)

        return ConverterReturn(annotations, metadata, None)

    @staticmethod
    def _read_labels(labels_file):
        """
        Extract label names from labels.txt file.
        """

        return read_txt(labels_file)

    def get_meta(self):
        '''
        Generate dataset metadata

        '''
        # read labels list
        labels = self._read_labels(get_path(self.data_dir, is_directory=True) / 'labels.txt')
        # create label map
        label_map = dict(enumerate(labels))
        return {'label_map': label_map}


    @staticmethod
    def _convert_annotations(test_dir, metadata, progress_callback, progress_interval):
        """
        Create annotation representations list.
        """

        # test directory contains files with names XXXX_class.png
        # we use regular expression to extract class names
        file_pattern_regex = re.compile(r'\d+_(\w+)\.png')
        labels = list(metadata['label_map'].values())
        annotations = []
        num_iterations = len(test_dir.glob('*.png'))
        # iterate over all png images in test directory
        for image_id, image in enumerate(test_dir.glob('*.png')):
            # get file name (e.g. from /foo/bar/image.png we get image.png)
            image_base = str(image.parts[-1])

            # extract class name from file name
            regex_match = re.match(file_pattern_regex, image_base)
            image_label = regex_match.group(1)

            # look up class index in label list
            class_id = labels.index(image_label)

            # create annotation representation object
            # Provided parameters can be differ depends on task.
            # ClassificationAnnotation contains image identifier and label for evaluation.
            annotations.append(ClassificationAnnotation(image_base, class_id))
            if progress_callback is not None and image_id % progress_interval == 0:
                progress_callback(image_id / num_iterations * 100)

        return annotations
