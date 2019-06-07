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

from pathlib import Path
import warnings

from ..representation import BrainTumorSegmentationAnnotation
from ..utils import get_path, read_txt, read_pickle
from ..config import StringField, PathField
from .format_converter import DirectoryBasedAnnotationConverter
from ..representation.segmentation_representation import GTMaskLoader


class BratsConverter(DirectoryBasedAnnotationConverter):
    __provider__ = 'brats'
    annotation_types = (BrainTumorSegmentationAnnotation, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'image_folder': StringField(optional=True, default='imagesTr', description="Image folder."),
            'mask_folder': StringField(optional=True, default='labelsTr', description="Mask folder.")
        })

        return parameters

    def configure(self):
        self.data_dir = self.get_value_from_config('data_dir')
        self.image_folder = self.get_value_from_config('image_folder')
        self.mask_folder = self.get_value_from_config('mask_folder')

    def convert(self):
        mask_folder = Path(self.mask_folder)
        image_folder = Path(self.image_folder)
        image_dir = get_path(self.data_dir / image_folder, is_directory=True)
        mask_dir = get_path(self.data_dir / mask_folder, is_directory=True)

        annotations = []
        for file_in_dir in image_dir.iterdir():
            file_name = file_in_dir.parts[-1]
            mask = mask_dir / file_name
            if not mask.exists():
                warnings.warn('Annotation mask for {} does not exists. File will be ignored.'.format(file_name))
                continue
            annotation = BrainTumorSegmentationAnnotation(
                str(image_folder / file_name),
                str(mask_folder / file_name),
            )

            annotations.append(annotation)

        return annotations


class BratsNumpyConverter(DirectoryBasedAnnotationConverter):
    __provider__ = 'brats_numpy'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'ids_file': PathField(description="Path to file, which contains names of images in dataset"),
            'labels_file': PathField(
                optional=True, default=None,
                description='Path to file, which contains labels (if omitted no labels will be shown)'
            ),
            'boxes_file': PathField(optional=True, default=None, description='Path to file with brain boxes'),
            'data_suffix': StringField(
                optional=True, default='_data_cropped', description='Suffix for files with data'
            ),
            'label_suffix': StringField(
                optional=True, default='_label_cropped', description='Suffix for files with ground truth data'
            )
        })

        return parameters

    def configure(self):
        self.data_dir = self.get_value_from_config('data_dir')
        self.ids_file = self.get_value_from_config('ids_file')
        self.labels_file = self.get_value_from_config('labels_file')
        self.boxes_file = self.get_value_from_config('boxes_file')
        self.data_suffix = self.get_value_from_config('data_suffix')
        self.label_suffix = self.get_value_from_config('label_suffix')

    def convert(self):
        ids = read_pickle(get_path(self.ids_file), encoding='latin1')
        boxes = read_pickle(get_path(self.boxes_file), encoding='latin1') if self.boxes_file else None

        annotations = []
        for i, name in enumerate(ids):
            data = name + self.data_suffix + '.npy'
            label = name + self.label_suffix + '.npy'

            if not self._check_files(data, label):
                warnings.warn('One of files {} or {} is not exist. Files will be ignored'.format(data, label))
                continue

            box = boxes[i, :, :] if self.boxes_file else None

            annotation = BrainTumorSegmentationAnnotation(
                data,
                label,
                GTMaskLoader.NUMPY,
                box
            )

            annotations.append(annotation)

        return annotations, self._get_meta()

    def _get_meta(self):
        if not self.labels_file:
            return None
        return {'label_map': [line for line in read_txt(self.labels_file)]}

    def _check_files(self, data, label):
        try:
            get_path(self.data_dir / data)
            get_path(self.data_dir / label)
            return True
        except (FileNotFoundError, IsADirectoryError):
            return False
