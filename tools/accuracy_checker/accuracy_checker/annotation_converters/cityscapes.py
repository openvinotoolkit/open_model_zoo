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

from pathlib import Path
from ..representation import SegmentationAnnotation
from ..representation.segmentation_representation import GTMaskLoader
from ..config import PathField, StringField, BoolField
from .format_converter import BaseFormatConverter, ConverterReturn, verify_label_map
from ..utils import check_file_existence, read_json


train_meta = {
    'label_map': {
        0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence', 5: 'pole', 6: 'traffic light',
        7: 'traffic sign', 8: 'vegetation', 9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car',
        14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle', 19: 'background'
    },
    'segmentation_colors': (
        (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
        (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0),
        (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32), (255, 255, 255)
    ),
    'background_label': 19
}

full_dataset_meta = {
    'segmentation_colors' : (
        (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), (81, 0, 81), (128, 64, 128),
        (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153),
        (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30),
        (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142),
        (0, 0, 70), (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32)
    ),
    'label_map': {
        0: 'unlabeled', 1:  'ego vehicle', 2: 'rectification border', 3: 'out of roi', 4: 'static', 5: 'dynamic',
        6: 'ground', 7: 'road', 8: 'sidewalk', 9: 'parking', 10: 'rail track', 11: 'building', 12: 'wall',
        13: 'fence', 14: 'guard rail', 15: 'bridge', 16: 'tunnel', 17: 'pole', 18: 'polegroup', 19: 'traffic light',
        20: 'traffic sign', 21: 'vegetation', 22: 'terrain', 23: 'sky', 24: 'person', 25: 'rider', 26: 'car',
        27: 'truck', 28: 'bus', 29: 'caravan', 30: 'trailer', 31: 'train', 32: 'motorcycle', 33: 'bicycle',
        -1: 'license plate'
    },
    'prediction_to_gt_labels': {
        0: 7, 1: 8, 2: 11, 3: 12, 4: 13, 5: 17, 6: 19, 7: 20, 8: 21, 9: 22,
        10: 23, 11: 24, 12: 25, 13: 26, 14: 27, 15: 28, 16: 31, 17: 32, 18: 33
    },
    'ignored_labels': [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1, 255]
}


class CityscapesConverter(BaseFormatConverter):
    __provider__ = 'cityscapes'
    annotation_types = (SegmentationAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'dataset_root_dir': PathField(is_directory=True, description="Path to dataset root."),
            'images_subfolder': StringField(
                optional=True,
                default='imgsFine/leftImg8bit/val',
                description="Path from dataset root to directory with validation images."
            ),
            'masks_subfolder': StringField(
                optional=True,
                default='gtFine/val',
                description="Path from dataset root to directory with ground truth masks."
            ),
            'masks_suffix': StringField(
                optional=True, default='_gtFine_labelTrainIds', description="Suffix for mask file names."
            ),
            'images_suffix': StringField(
                optional=True, default='_leftImg8bit', description="Suffix for image file names."
            ),
            'use_full_label_map': BoolField(
                optional=True,
                default=False,
                description="Allows to use full label map with 33 classes instead train label map with 18 classes."
            ),
            'dataset_meta_file': PathField(
                description='path to json file with dataset meta (e.g. label_map, color_encoding', optional=True
            )
        })

        return configuration_parameters

    def configure(self):
        self.dataset_root = self.get_value_from_config('dataset_root_dir')
        self.images_dir = self.get_value_from_config('images_subfolder')
        self.masks_dir = self.get_value_from_config('masks_subfolder')
        self.masks_suffix = self.get_value_from_config('masks_suffix')
        self.images_suffix = self.get_value_from_config('images_suffix')
        self.use_full_label_map = self.get_value_from_config('use_full_label_map')
        self.dataset_meta_file = self.get_value_from_config('dataset_meta_file')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        images = list(self.dataset_root.rglob(r'{}/*/*{}.png'.format(self.images_dir, self.images_suffix)))
        content_errors = None if not check_content else []
        annotations = []
        num_iterations = len(images)
        for idx, image in enumerate(images):
            identifier = str(Path(self.images_dir).joinpath(*image.parts[-2:]))
            mask = str(Path(self.masks_dir) / image.parts[-2] / self.masks_suffix.join(
                str(image.name).split(self.images_suffix)
            ))
            if check_content:
                if not check_file_existence(self.dataset_root / mask):
                    content_errors.append('{}: does not exist'.format(self.dataset_root / mask))
            annotations.append(SegmentationAnnotation(identifier, mask, mask_loader=GTMaskLoader.PILLOW))
            if progress_callback is not None and idx % progress_interval == 0:
                progress_callback(idx / num_iterations * 100)

        return ConverterReturn(annotations, self.generate_meta(), content_errors)

    def generate_meta(self):
        if self.dataset_meta_file is not None:
            meta = read_json(self.dataset_meta_file)
            if 'label_map' in meta:
                meta['label_map'] = verify_label_map(meta['label_map'])
            if 'labels' in meta and 'label_map' not in meta:
                meta['label_map'] = dict(enumerate(meta['labels']))
            return meta
        return full_dataset_meta if self.use_full_label_map else train_meta
