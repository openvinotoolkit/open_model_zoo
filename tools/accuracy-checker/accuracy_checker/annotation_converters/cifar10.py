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
import os
from PIL import Image
import numpy as np
from ..config import PathField, BoolField
from ..representation import ClassificationAnnotation
from ..utils import read_pickle

from .format_converter import BaseFormatConverter

CIFAR10_LABELS_LIST = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


class Cifar10FormatConverter(BaseFormatConverter):
    """
    cifar10 dataset converter. All annotation converters should be derived from BaseFormatConverter class.
    """

    # register name for this converter
    # this name will be used for converter class look up
    __provider__ = 'cifar10'
    annotation_types = (ClassificationAnnotation, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'data_batch_file': PathField(description="Path to pickle file which contain dataset batch."),
            'convert_images': BoolField(
                optional=True,
                default=False,
                description="Allows to convert images from pickle file to user specified directory."
            ),
            'converted_images_dir': PathField(
                optional=True, is_directory=True, check_exists=False, description="Path to converted images location."
            ),
            'has_background': BoolField(
                optional=True,
                default=False,
                description="Allows to add background label to original labels and convert dataset "
                            "for 11 classes instead 10"
            )
        })

        return parameters

    def configure(self):
        """
        This method is responsible for obtaining the necessary parameters
        for converting from the command line or config.
        """
        self.data_batch_file = self.get_value_from_config('data_batch_file')
        self.has_background = self.get_value_from_config('has_background')
        self.converted_images_dir = self.get_value_from_config('converted_images_dir')
        if not self.converted_images_dir:
            self.converted_images_dir = os.path.join(self.data_batch_file.parent, 'converted_images')
        self.convert_images = self.get_value_from_config('convert_images')

    def convert(self):
        """
        This method is executed automatically when convert.py is started.
        All arguments are automatically got from command line arguments or config file in method configure

        Returns:
            annotations: list of annotation representation objects.
            meta: dictionary with additional dataset level metadata.
        """
        # create directory for storing images if it is necessary
        if self.convert_images and not self.converted_images_dir.exists():
            self.converted_images_dir.mkdir(parents=True)

        annotation = []
        # read original dataset annotation
        annotation_dict = read_pickle(self.data_batch_file, encoding='latin1')
        labels = annotation_dict['labels']
        images = annotation_dict['data']
        images = images.reshape(images.shape[0], 3, 32, 32).astype(np.uint8)
        image_file = '{}_{}.png'
        # Originally dataset labels start from 0, some networks can be trained with usage 1 as label start.
        labels_offset = 0 if not self.has_background else 1

        # convert each annotation object to ClassificationAnnotation
        for data_id, (label, feature) in enumerate(zip(labels, images)):
            # generate id of image which will be used for evaluation (usually name of file is used)
            # file name represented as {id}_{class}.png, where id is index of image in dataset,
            # label is text description of dataset class e.g. 1_cat.png
            identifier = image_file.format(data_id, CIFAR10_LABELS_LIST[label])
            # Create representation for image. Provided parameters can be differ depends on task.
            # ClassificationAnnotation contains image identifier and label for evaluation.
            annotation.append(ClassificationAnnotation(identifier, label + labels_offset))
            # if it is necessary convert dataset image to png format and store it to converted_images_dir folder.
            if self.convert_images:
                image = Image.fromarray(np.transpose(feature, (1, 2, 0)))
                image = image.convert('RGB')
                image.save(str(self.converted_images_dir / identifier))

        # crete metadata for dataset. Provided additional information is task specific and can includes, for example
        # label_map, information about background, used class color representation (for semantic segmentation task)
        # If your dataset does not have additional meta, you can to not provide it.
        meta = {
            'label_map': {
                label_id + labels_offset: label_name for label_id, label_name in enumerate(CIFAR10_LABELS_LIST)
            }
        }

        if self.has_background:
            meta['label_map'][0] = 'background'
            meta['background_label'] = 0

        return annotation, meta
