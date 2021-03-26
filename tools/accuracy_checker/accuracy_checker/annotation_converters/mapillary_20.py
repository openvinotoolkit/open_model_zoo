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

from ..config import PathField, StringField, ConfigError
from ..representation import SegmentationAnnotation
from ..representation.segmentation_representation import GTMaskLoader
from ..utils import get_path
from .format_converter import BaseFormatConverter, ConverterReturn


class MapillaryBaseConverter(BaseFormatConverter):
    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'data_dir': PathField(
                optional=True, is_directory=True,
                description="Path to dataset root folder. Relative paths to images and masks directory "
                            "determine as imgs and masks respectively. "
                            "In way when images and masks are located in non default directories, "
                            "you can use parameters described below."
            ),
            'images_dir': PathField(
                optional=True, is_directory=True, description="Path to images folder."
            ),
            'mask_dir': PathField(
                optional=True, is_directory=True, description="Path to ground truth mask folder."
            ),
            'images_subfolder': StringField(
                optional=True, default='imgs', description="Sub-directory for images."
            ),
            'mask_subfolder': StringField(
                optional=True, default='masks', description="Sub-directory for ground truth mask."
            )
        })

        return parameters

    def configure(self):
        data_dir = self.get_value_from_config('data_dir')
        images_dir = self.get_value_from_config('images_dir')
        mask_dir = self.get_value_from_config('mask_dir')
        images_folder = self.get_value_from_config('images_subfolder')
        mask_folder = self.get_value_from_config('mask_subfolder')

        if not data_dir and (not images_dir or not mask_dir):
            raise ConfigError('data_dir or images_dir and mask_dir should be provided for conversion')

        images_path = images_dir if images_dir else data_dir / images_folder
        mask_path = mask_dir if mask_dir else data_dir / mask_folder

        self.images_dir = get_path(images_path, is_directory=True)
        self.mask_dir = get_path(mask_path, is_directory=True)

    def convert(self, *args, **kwargs):
        pass


class Mapillary20Converter(MapillaryBaseConverter):
    __provider__ = 'mapillary_20'
    annotation_types = (SegmentationAnnotation, )

    label_map = {
        0: 'Road',
        1: 'Sidewalk',
        2: 'Building',
        3: 'Wall',
        4: 'Fence',
        5: 'Pole',
        6: 'Traffic Light',
        7: 'Traffic Sign',
        8: 'Vegetation',
        9: 'Terrain',
        10: 'Sky',
        11: 'Person',
        12: 'Rider',
        13: 'Car',
        14: 'Truck',
        15: 'Bus',
        16: 'Train',
        17: 'Motorcycle',
        18: 'Bicycle',
        19: 'Ego-Vehicle'
    }

    def convert(self, *args, **kwargs):
        annotations = []
        for file_in_dir in self.images_dir.glob('*.png'):
            annotation = SegmentationAnnotation(file_in_dir.name, file_in_dir.name, mask_loader=GTMaskLoader.PILLOW)
            annotations.append(annotation)

        return ConverterReturn(annotations, {'label_map': self.label_map}, None)


class MapillaryVistasConverter(MapillaryBaseConverter):
    __provider__ = 'mapillary_vistas'
    annotation_types = (SegmentationAnnotation, )

    meta = {
        'label_map': {
            0: 'Unlabeled',
            1: 'Road',
            2: 'Sidewalk',
            3: 'Building',
            4: 'Wall',
            5: 'Fence',
            6: 'Pole',
            7: 'Traffic Light',
            8: 'Traffic Sign',
            9: 'Vegetation',
            10: 'Terrain',
            11: 'Sky',
            12: 'Person',
            13: 'Rider',
            14: 'Car',
            15: 'Truck',
            16: 'Bus',
            17: 'Train',
            18: 'Motorcycle',
            19: 'Bicycle'
        },
        'background_label': 0,
        'segmentation_colors': (
            (0, 0, 0), (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
            (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0),
            (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)
        )
    }

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'images_subfolder': StringField(
                optional=True, default='images', description="Sub-directory for images."
            ),
            'mask_subfolder': StringField(
                optional=True, default='labels', description="Sub-directory for ground truth mask."
            )
        })

        return parameters

    def convert(self, *args, **kwargs):
        annotations = []
        for file_in_dir in self.mask_dir.glob('*.png'):
            identifier = file_in_dir.name.replace('.png', '.jpg')
            annotation = SegmentationAnnotation(identifier, file_in_dir.name, GTMaskLoader.PILLOW_CONVERT_TO_RGB)
            annotations.append(annotation)

        return ConverterReturn(annotations, self.meta, None)
