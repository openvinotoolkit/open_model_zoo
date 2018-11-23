"""
 Copyright (c) 2018 Intel Corporation

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
import pathlib
import re

from accuracy_checker.representation import ClassificationAnnotation

from .format_converter import BaseFormatConverter

# test directory contains files with names XXXX_class.png
# we use regular expression to extract class names
FILE_PATTERN_REGEX = re.compile(r"\d+_(\w+)\.png")


class SampleConverter(BaseFormatConverter):
    """
    Sample dataset converter. All annotation converters should be derived from BaseFormatConverter class.
    """

    # register name for this converter
    # this name will be used for converter class look up
    __provider__ = "sample"

    def convert(self, dataset_directory: str):
        """
        This method is executed automatically when convert.py is started.
        All arguments are automatically forwarded from command line arguments

        Args:
            dataset_directory: path to sample dataset

        Returns:
            annotations: list of annotation representation objects.
            meta: dictionary with additional dataset level metadata

        """
        # convert path string to Path object
        dataset_directory = pathlib.Path(dataset_directory)

        # read and convert annotation
        labels = self._read_labels(dataset_directory / 'labels.txt')
        annotations = self._convert_annotations(dataset_directory / 'test', labels)

        # convert label list to label map
        label_map = {i: labels[i] for i in range(len(labels))}
        metadata = {'label_map': label_map}

        return annotations, metadata

    @staticmethod
    def _read_labels(labels_file):
        """Extract label names from labels.txt file"""
        with labels_file.open() as f:
            lines = f.read().split()

        return lines

    @staticmethod
    def _convert_annotations(test_dir, labels):
        """Create annotation representations list"""
        annotations = []

        # iterate over all png images in test directory
        for image in test_dir.glob("*.png"):
            # get file name (e.g. from /foo/bar/image.png we get image.png)
            image_base = str(image.parts[-1])

            # extract class name from file name
            regex_match = re.match(FILE_PATTERN_REGEX, image_base)
            image_label = regex_match.group(1)

            # look up class index in label list
            class_id = labels.index(image_label)

            # create annotation representation object
            annotations.append(ClassificationAnnotation(image_base, class_id))

        return annotations
