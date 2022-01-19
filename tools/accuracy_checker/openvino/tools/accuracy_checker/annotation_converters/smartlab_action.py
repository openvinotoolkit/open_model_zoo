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

import re
import os
from ..config import PathField
from ..representation import ClassificationAnnotation
from ..utils import get_path, read_txt

from .format_converter import BaseFormatConverter, ConverterReturn

class SmartlabActionConverter(BaseFormatConverter):
    """
    Sample dataset converter. All annotation converters should be derived from BaseFormatConverter class.
    """

    # register name for this converter
    # this name will be used for converter class look up
    __provider__ = 'smartlab_action'
    annotation_types = (ClassificationAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'images_dir': PathField(is_directory=True, description="Path to sample dataset root directory."),
            'annotation_txt': PathField(is_directory=False, description="Path to sample annoation file.")
        })

        return configuration_parameters

    def configure(self):
        """
        This method is responsible for obtaining the necessary parameters
        for converting from the command line or config.
        """
        self.images_dir = self.config['images_dir']
        self.annotation_txt = self.config['annotation_txt']

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        """
        This method is executed automatically when convert.py is started.
        All arguments are automatically got from command line arguments or config file in method configure

        Returns:
            annotations: list of annotation representation objects.
            meta: dictionary with additional dataset level metadata (if provided)
        """

        # high_directory = get_path(os.path.join(self.images_dir, 'high'), is_directory=True)
        # top_directory = get_path(os.path.join(self.images_dir, 'top'), is_directory=True)

        # create dataset metadata
        metadata = self.get_meta()
        # print(metadata)

        # # read and convert annotation
        # high_dir = high_directory
        # top_dir = top_directory

        # we use regular expression to extract class names
        labels = list(metadata['label_map'].values())

        annotations = []
        num_iterations = len(labels)
        # iterate over all jpg images in test directory
        for image_id, line in enumerate(labels):
            image_base, class_id = line.split(' ')
            # print(image_id, image_base, class_id)

            # ClassificationAnnotation contains image identifier and label for evaluation.
            annotations.append(ClassificationAnnotation(image_base, int(class_id)))
            # print('=== annotation ====')
            # print(annotations[-1].label) # 11
            # print(annotations[-1].identifier) # frame04443.jpg

            if progress_callback is not None and image_id % progress_interval == 0:
                progress_callback(image_id / num_iterations * 100)

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
        labels = self._read_labels(self.annotation_txt)
        # create label map
        label_map = dict(enumerate(labels))
        return {'label_map': label_map}