"""
Copyright (c) 2018-2020 Intel Corporation

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

import numpy as np
from PIL import Image
from ..config import PathField, BoolField, NumberField
from ..representation import ClassificationAnnotation
from ..utils import read_pickle, check_file_existence, read_json

from .format_converter import BaseFormatConverter, ConverterReturn, verify_label_map


CIFAR10_LABELS_LIST = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

CIFAR100_LABELS_LIST = [
    'beaver', 'dolphin', 'otter', 'seal', 'whale',
    'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
    'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
    'bottles', 'bowls', 'cans', 'cups', 'plates',
    'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
    'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
    'bed', 'chair', 'couch', 'table', 'wardrobe',
    'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
    'bear', 'leopard', 'lion', 'tiger', 'wolf',
    'bridge', 'castle', 'house', 'road', 'skyscraper',
    'cloud', 'forest', 'mountain', 'plain', 'sea',
    'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
    'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
    'crab', 'lobster', 'snail', 'spider', 'worm',
    'baby', 'boy', 'girl', 'man', 'woman',
    'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
    'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
    'maple', 'oak', 'palm', 'pine', 'willow',
    'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
    'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor'
]

class_map = {
    10: (CIFAR10_LABELS_LIST, 'labels'),
    100: (CIFAR100_LABELS_LIST, 'fine_labels')
}


class CifarFormatConverter(BaseFormatConverter):
    """
    cifar10 dataset converter. All annotation converters should be derived from BaseFormatConverter class.
    """

    # register name for this converter
    # this name will be used for converter class look up
    __provider__ = 'cifar'
    annotation_types = (ClassificationAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
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
            ),
            'dataset_meta_file': PathField(
                description='path to json file with dataset meta (e.g. label_map, color_encoding', optional=True
            ),
            'num_classes': NumberField(
                optional=True, default=10, value_type=int,
                description='the number of classes in the dataset without background (10 or 100)'

            )
        })

        return configuration_parameters

    def configure(self):
        """
        This method is responsible for obtaining the necessary parameters
        for converting from the command line or config.
        """
        self.data_batch_file = self.get_value_from_config('data_batch_file')
        self.has_background = self.get_value_from_config('has_background')
        self.num_classes = self.get_value_from_config('num_classes')
        self.converted_images_dir = self.get_value_from_config('converted_images_dir')
        if not self.converted_images_dir:
            self.converted_images_dir = self.data_batch_file.parent / 'converted_images'
        self.convert_images = self.get_value_from_config('convert_images')
        self.dataset_meta = self.get_value_from_config('dataset_meta_file')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
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
        check_images = check_content and not self.convert_images
        content_errors = None

        if self.converted_images_dir and check_content:
            if not self.converted_images_dir.exists():
                content_errors = ['{}: does not exist'.format(self.converted_images_dir)]
                check_images = False

        if check_images:
            content_errors = []

        annotation = []
        # read original dataset annotation
        annotation_dict = read_pickle(self.data_batch_file, encoding='latin1')
        # Originally dataset labels start from 0, some networks can be trained with usage 1 as label start.
        labels_offset = 0 if not self.has_background else 1
        # crete metadata for dataset. Provided additional information is task specific and can includes, for example
        # label_map, information about background, used class color representation (for semantic segmentation task)
        # If your dataset does not have additional meta, you can to not provide it.
        meta, label_names, labels_id = self.generate_meta(labels_offset)
        labels = annotation_dict[labels_id]
        images = annotation_dict['data']
        images = images.reshape(images.shape[0], 3, 32, 32).astype(np.uint8)
        image_file = '{}_{}.png'
        num_iterations = len(labels)
        # convert each annotation object to ClassificationAnnotation
        for data_id, (label, feature) in enumerate(zip(labels, images)):
            # generate id of image which will be used for evaluation (usually name of file is used)
            # file name represented as {id}_{class}.png, where id is index of image in dataset,
            # label is text description of dataset class e.g. 1_cat.png
            identifier = image_file.format(data_id, label_names[label] if label_names else label)
            # Create representation for image. Provided parameters can be differ depends on task.
            # ClassificationAnnotation contains image identifier and label for evaluation.
            annotation.append(ClassificationAnnotation(identifier, label + labels_offset))
            # if it is necessary convert dataset image to png format and store it to converted_images_dir folder.
            if self.convert_images:
                image = Image.fromarray(np.transpose(feature, (1, 2, 0)))
                image = image.convert('RGB')
                image.save(str(self.converted_images_dir / identifier))

            # check image existence if it is necessary
            if check_images:
                if not check_file_existence(self.converted_images_dir / identifier):
                    # add error to errors list if file not found
                    content_errors.append('{}: does not exist'.format(self.converted_images_dir / identifier))

            if progress_callback is not None and data_id % progress_interval == 0:
                progress_callback(data_id / num_iterations * 100)

        return ConverterReturn(annotation, meta, content_errors)

    def generate_meta(self, labels_offset):
        labels, labels_id = class_map.get(self.num_classes, ([], 'labels'))
        meta = {}
        if self.dataset_meta:
            meta = read_json(self.dataset_meta)
            if 'label_map' in meta:
                meta['label_map'] = verify_label_map(meta['label_map'])
                labels = list(meta['label_map'].values())
                return meta, labels, labels_id
            labels = meta.get('labels', labels)
        meta.update({'label_map': {label_id + labels_offset: label_name for label_id, label_name in enumerate(labels)}})

        if self.has_background:
            meta['label_map'][0] = 'background'
            meta['background_label'] = 0

        return meta, labels, labels_id
