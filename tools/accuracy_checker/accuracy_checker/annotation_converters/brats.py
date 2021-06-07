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

from pathlib import Path
import warnings
import re

from ..representation import BrainTumorSegmentationAnnotation, NiftiRegressionAnnotation
from ..utils import get_path, read_txt, read_pickle, check_file_existence, UnsupportedPackage
from ..config import StringField, PathField, BoolField, NumberField
from .format_converter import DirectoryBasedAnnotationConverter
from ..representation.segmentation_representation import GTMaskLoader
from .format_converter import ConverterReturn

try:
    import nibabel as nib
except ImportError as import_error:
    nib = UnsupportedPackage("nibabel", import_error.msg)

class BratsConverter(DirectoryBasedAnnotationConverter):
    __provider__ = 'brats'
    annotation_types = (BrainTumorSegmentationAnnotation, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'image_folder': StringField(optional=True, default='imagesTr', description="Image folder."),
            'mask_folder': StringField(optional=True, default='labelsTr', description="Mask folder."),
            'labels_file': PathField(optional=True, default=None, description="File with labels"),
            'relaxed_names': BoolField(optional=True, default=False),
            'frame_separator': StringField(optional=True, default='#',
                                           description="Separator between filename and frame number"),
            'multi_frame': BoolField(optional=True, default=False,
                                     description="Add annotation for each frame in source file"),
            'frame_axis': NumberField(optional=True, default=-1, description="Frames dimension axis"),
            'as_regression': BoolField(optional=True, default=False,
                                       description="annotate dataset as RegressionAnnotation"),
            'mask_channels_first': BoolField(optional=True, default=False)
        })

        return parameters

    def configure(self):
        self.data_dir = self.get_value_from_config('data_dir')
        self.image_folder = self.get_value_from_config('image_folder')
        self.mask_folder = self.get_value_from_config('mask_folder')
        self.labels_file = self.get_value_from_config('labels_file')
        self.mask_channels_first = self.get_value_from_config('mask_channels_first')
        self.relaxed_names = self.get_value_from_config('relaxed_names')
        self.multi_frame = self.get_value_from_config('multi_frame')
        self.frame_axis = int(self.get_value_from_config('frame_axis'))
        self.frame_separator = self.get_value_from_config('frame_separator')
        self.as_regression = self.get_value_from_config('as_regression')

    def convert(self, check_content=False, **kwargs):
        if self.multi_frame and isinstance(nib, UnsupportedPackage):
            nib.raise_error(self.__provider__)
        if self.as_regression:
            annotation_class = NiftiRegressionAnnotation
            reader_config = {'to_4D': False, 'multi_frame': self.multi_frame,
                             'separator': self.frame_separator, 'frame_axis': self.frame_axis}
        else:
            annotation_class = BrainTumorSegmentationAnnotation
            reader_config = {'loader': GTMaskLoader.NIFTI_CHANNELS_FIRST if self.mask_channels_first else
                                       GTMaskLoader.NIFTI}

        mask_folder = Path(self.mask_folder)
        image_folder = Path(self.image_folder)
        image_dir = get_path(self.data_dir / image_folder, is_directory=True)
        mask_dir = get_path(self.data_dir / mask_folder, is_directory=True)
        content_check_errors = [] if check_content else None

        annotations = []
        for file_in_dir in image_dir.iterdir():
            file_name = file_in_dir.parts[-1]
            mask_file_name = (
                file_name.rsplit('.', 1)[0] + '.nii.gz' if not file_name.endswith('.nii.gz') else file_name
            )
            if not self.relaxed_names:
                mask = mask_dir / mask_file_name
            else:
                mask = self.find_file_by_id(mask_dir, mask_file_name)
                mask_file_name = mask.parts[-1]
            if not mask.exists():
                if not check_content:
                    warnings.warn('Annotation mask for {} does not exist. File will be ignored.'.format(file_name))
                else:
                    content_check_errors.append(
                        '{}: annotation mask does not exist, please remove this file or add gt mask ({})'.format(
                            file_in_dir, mask)
                    )
                continue

            if self.multi_frame:
                nib_image = nib.load(file_in_dir)
                total_frames = nib_image.dataobj.shape[self.frame_axis]
                for frame_cnt in range(total_frames):
                    annotation = annotation_class(
                        "{}{}{}".format(str(image_folder / file_name), self.frame_separator, frame_cnt),
                        "{}{}{}".format(str(mask_folder / mask_file_name), self.frame_separator, frame_cnt),
                        **reader_config
                    )
                    annotations.append(annotation)
            else:

                annotation = annotation_class(
                    str(image_folder / file_name),
                    str(mask_folder / mask_file_name),
                    **reader_config
                )

                annotations.append(annotation)

        return ConverterReturn(annotations, self._get_meta(), content_check_errors)

    def _get_meta(self):
        if not self.labels_file:
            return None
        return {'label_map': dict(enumerate(read_txt(self.labels_file)))}

    @staticmethod
    def find_file_by_id(search_dir, file_name):
        def get_index(file_name):
            numbers = [int(s) for s in re.findall(r'\d+', file_name)]
            if not numbers:
                raise ValueError('no numeric in {}'.format(file_name))
            return numbers

        idx = get_index(file_name)
        found_files = []
        for i in idx:
            found_files.extend(search_dir.glob('*{}*'.format(i)))
        if not found_files:
            return None
        return found_files[0]

    @staticmethod
    def find_file_with_prefix(search_dir, file_name):
        found_files = list(search_dir.glob('*{}*'.format(file_name)))
        return found_files[0] if found_files else None


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

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        ids = read_pickle(get_path(self.ids_file), encoding='latin1')
        boxes = read_pickle(get_path(self.boxes_file), encoding='latin1') if self.boxes_file else None
        check_content_errors = [] if check_content else None

        annotations = []
        num_iterations = len(ids)
        for i, name in enumerate(ids):
            data = name + self.data_suffix + '.npy'
            label = name + self.label_suffix + '.npy'
            files_exists = True

            if not check_file_existence(self.data_dir / data):
                warning_message = '{}: does not exist'.format(self.data_dir / data)
                warnings.warn(warning_message)
                if check_content:
                    check_content_errors.append(warning_message)
                files_exists = False

            if not check_file_existence(self.data_dir / label):
                warning_message = '{}: does not exist'.format(self.data_dir / label)
                warnings.warn(warning_message)
                if check_content:
                    check_content_errors.append(warning_message)
                files_exists = False

            if not files_exists:
                continue

            box = boxes[i, :, :] if self.boxes_file else None

            annotation = BrainTumorSegmentationAnnotation(data, label, GTMaskLoader.NUMPY, box)

            annotations.append(annotation)
            if progress_callback is not None and i % progress_interval == 0:
                progress_callback(i / num_iterations * 100)

        return ConverterReturn(annotations, self._get_meta(), check_content_errors)

    def _get_meta(self):
        if not self.labels_file:
            return None
        return {'label_map': dict(enumerate(read_txt(self.labels_file)))}
