"""
Copyright (c) 2018-2024 Intel Corporation

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

import cv2
import numpy as np
from .format_converter import BaseFormatConverter, ConverterReturn
from ..representation import ClassificationAnnotation, AnomalySegmentationAnnotation
from ..config import PathField, BoolField
from ..utils import get_path, check_file_existence


class MVTecDatasetConverter(BaseFormatConverter):
    __provider__ = 'mvtec'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'data_dir': PathField(is_directory=True, optional=False, description='Dataset root dir'),
            'classification_only': BoolField(optional=True, default=False),

        })
        return params

    def configure(self):
        self.data_dir = self.get_value_from_config('data_dir')
        self.test_images_dir = get_path(self.data_dir / 'test', is_directory=True)
        self.classification_only = self.get_value_from_config('classification_only')
        if not self.classification_only:
            self.reference_mask_dir = self.data_dir / 'ground_truth'

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        test_images = list(self.test_images_dir.rglob("**/*.png"))
        num_iterations = len(test_images)
        annotations = []
        errors = [] if check_content else None
        for idx, image in enumerate(test_images):
            label = image.parent.name
            label_id = 0 if label == 'good' else 1
            identifier = str(image.relative_to(self.test_images_dir))
            if self.classification_only:
                annotation = ClassificationAnnotation(identifier, label_id)
                annotations.append(annotation)
            else:
                mask = None
                mask_path = None
                if not label_id:
                    img = cv2.imread(str(image))
                    h, w = img.shape[:2]
                    mask = np.zeros((h, w), dtype=np.uint8)
                else:
                    mask_path = str(
                        image.with_name(image.stem + '_mask' + image.suffix).relative_to(self.test_images_dir))
                annotation = AnomalySegmentationAnnotation(identifier, mask_path, label_id)
                if mask_path is None:
                    annotation.mask = mask
                if check_content and mask_path:
                    if not check_file_existence(self.reference_mask_dir / mask_path):
                        errors.append('{}: does not exist'.format(self.reference_mask_dir / mask_path))
                annotations.append(annotation)
            if progress_callback and idx % progress_interval == 0:
                progress_callback(idx * 100 / num_iterations)
        return ConverterReturn(annotations, self.get_meta(), errors)

    @staticmethod
    def get_meta():
        return {'label_map': {0: 'good', 1: 'defect'}}
