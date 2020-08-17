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

import re
import os
import warnings
from pathlib import Path
import cv2
import numpy as np
from ..config import PathField, StringField, BoolField, ConfigError, NumberField, DictField
from ..representation import SuperResolutionAnnotation, ContainerAnnotation
from ..representation.image_processing import GTLoader
from ..utils import check_file_existence
from ..data_readers import MultiFramesInputIdentifier
from .format_converter import BaseFormatConverter, ConverterReturn

LOADERS_MAPPING = {
    'opencv': GTLoader.OPENCV,
    'pillow': GTLoader.PILLOW,
    'dicom': GTLoader.DICOM
}


class SRConverter(BaseFormatConverter):
    __provider__ = 'super_resolution'
    annotation_types = (SuperResolutionAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'data_dir': PathField(
                is_directory=True, optional=True,
                description="Path to folder, where images in low and high resolution are located."
            ),
            'lr_dir': PathField(
                is_directory=True, optional=True,
                description="Path to directory, where images in low resolution are located."
            ),
            'hr_dir': PathField(
                is_directory=True, optional=True,
                description="Path to directory, where images in high resolution are located."
            ),
            'upsampled_dir': PathField(
                is_directory=True, optional=True,
                description="Path to directory, where upsampled images are located, if 2 streams used."
            ),
            'lr_suffix': StringField(
                optional=True, default="lr", description="Low resolution file name's suffix."
            ),
            'hr_suffix': StringField(
                optional=True, default="hr", description="High resolution file name's suffix."
            ),
            'two_streams': BoolField(optional=True, default=False, description="2 streams is used"),
            'annotation_loader': StringField(
                optional=True, choices=LOADERS_MAPPING.keys(), default='pillow',
                description="Which library will be used for ground truth image reading. "
                            "Supported: {}".format(', '.join(LOADERS_MAPPING.keys()))
            ),
            'upsample_suffix': StringField(
                optional=True, default='upsample', description='Upsampled file name`s suffix, if 2 streams used.'
            ),
            'generate_upsample': BoolField(
                optional=True, default=False, description='allows to generate bicubic interpolation for images'
            ),
            'upsample_factor': NumberField(
                optional=True, default=4, value_type=int, min_value=1,
                description='upsampling factor if generation upsample used.'
            )
        })

        return configuration_parameters

    def configure(self):
        self.data_dir = self.get_value_from_config('data_dir')
        self.lr_dir = self.get_value_from_config('lr_dir') or self.data_dir
        if not self.lr_dir:
            raise ConfigError('One of the parameters: data_dir or lr_dir should be provided for conversion')
        self.hr_dir = self.get_value_from_config('hr_dir') or self.data_dir
        self.lr_suffix = self.get_value_from_config('lr_suffix')
        self.hr_suffix = self.get_value_from_config('hr_suffix')
        self.upsample_suffix = self.get_value_from_config('upsample_suffix')
        self.two_streams = self.get_value_from_config('two_streams')
        if self.two_streams:
            self.upsampled_dir = self.get_value_from_config('upsampled_dir')
            if not self.upsampled_dir:
                self.upsampled_dir = self.lr_dir

            if self.data_dir:
                try:
                    self.lr_dir.relative_to(self.data_dir)
                    self.upsampled_dir.relative_to(self.data_dir)
                except:
                    raise ConfigError('data_dir parameter should be provided for conversion as common part of paths '
                                      'lr_dir and upsampled_dir, if 2 streams used')
        self.annotation_loader = LOADERS_MAPPING.get(self.get_value_from_config('annotation_loader'))
        if not self.annotation_loader:
            raise ConfigError('provided not existing loader')
        self.generate_upsample = self.get_value_from_config('generate_upsample')
        self.upsample_factor = self.get_value_from_config('upsample_factor')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        content_errors = [] if check_content else None
        file_list_lr = []
        for file_in_dir in self.lr_dir.iterdir():
            if self.lr_suffix in file_in_dir.parts[-1]:
                file_list_lr.append(file_in_dir)

        annotation = []
        num_iterations = len(file_list_lr)
        for lr_id, lr_file in enumerate(file_list_lr):
            lr_file_name = lr_file.parts[-1]
            upsampled_file_name = self.upsample_suffix.join(lr_file_name.split(self.lr_suffix))
            if self.two_streams and self.generate_upsample:
                self.generate_upsample_file(lr_file, self.upsample_factor, upsampled_file_name, self.upsampled_dir)
            hr_file_name = self.hr_suffix.join(lr_file_name.split(self.lr_suffix))
            if check_content:
                if not self.hr_dir:
                    content_errors.append('No one of the data_dir or hr_dir parameters are provided')
                if self.hr_dir and not check_file_existence(self.hr_dir / hr_file_name):
                    content_errors.append('{}: does not exist'.format(self.hr_dir / hr_file_name))
                if self.two_streams and not check_file_existence(self.upsampled_dir / upsampled_file_name):
                    content_errors.append('{}: does not exist'.format(self.upsampled_dir / upsampled_file_name))
            identifier = lr_file_name
            if self.two_streams:
                relative_dir = self.data_dir or os.path.commonpath([self.lr_dir, self.upsampled_dir])

                if self.lr_dir != self.upsampled_dir:
                    warnings.warn("lr_dir and upsampled_dir are different folders."
                                  "Make sure that data_source is {}".format(relative_dir))

                identifier = [str(lr_file.relative_to(relative_dir)),
                              str(Path(self.upsampled_dir, upsampled_file_name).relative_to(relative_dir))]
            annotation.append(SuperResolutionAnnotation(identifier, hr_file_name, gt_loader=self.annotation_loader))
            if progress_callback is not None and lr_id % progress_interval == 0:
                progress_callback(lr_id / num_iterations * 100)

        return ConverterReturn(annotation, None, content_errors)

    @staticmethod
    def generate_upsample_file(original_image_path, scale_factor, upsampled_file_name, upsampled_image_path):
        image = cv2.imread(str(original_image_path))
        upsampled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(upsampled_image_path / upsampled_file_name), upsampled_image)


class SRMultiFrameConverter(BaseFormatConverter):
    __provider__ = 'multi_frame_super_resolution'
    annotation_types = (SuperResolutionAnnotation, )

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'data_dir': PathField(
                is_directory=True, description="Path to folder, where images in low and high resolution are located."
            ),
            'lr_suffix': StringField(
                optional=True, default="lr", description="Low resolution file name's suffix."
            ),
            'hr_suffix': StringField(
                optional=True, default="hr", description="High resolution file name's suffix."
            ),
            'number_input_frames': NumberField(
                description='number inputs per inference', value_type=int,
            ),
            'annotation_loader': StringField(
                optional=True, choices=LOADERS_MAPPING.keys(), default='pillow',
                description="Which library will be used for ground truth image reading. "
                            "Supported: {}".format(', '.join(LOADERS_MAPPING.keys()))
            )
        })
        return params

    def configure(self):
        self.data_dir = self.get_value_from_config('data_dir')
        self.lr_suffix = self.get_value_from_config('lr_suffix')
        self.hr_suffix = self.get_value_from_config('hr_suffix')
        self.annotation_loader = LOADERS_MAPPING.get(self.get_value_from_config('annotation_loader'))
        self.num_frames = self.get_value_from_config('number_input_frames')
        self.max_frame_id = self.get_value_from_config('max_frame_id')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        def get_index(image_name, suffix):
            name_parts = image_name.split(suffix)
            numbers = []
            for part in name_parts:
                numbers += [int(s) for s in re.findall(r'\d+', part)]
            if not numbers:
                raise ValueError('no numeric in {}'.format(image_name))
            return numbers[-1]

        content_errors = [] if check_content else None
        frames_ids = []
        frame_names = []
        annotations = []
        for file_in_dir in self.data_dir.iterdir():
            if file_in_dir.suffix not in ['.jpg', '.png']:
                continue
            image_name = file_in_dir.parts[-1]
            if self.lr_suffix in image_name and self.hr_suffix not in image_name:
                frame_names.append(image_name)
                frames_ids.append(get_index(image_name, self.lr_suffix))
        sorted_frames = np.argsort(frames_ids)
        frames_ids.sort()
        sorted_frame_names = [frame_names[idx] for idx in sorted_frames]

        num_iterations = len(frames_ids)
        for idx, _ in enumerate(frames_ids):
            if len(frames_ids) - idx < self.num_frames:
                break
            input_ids = list(range(self.num_frames))
            input_frames = [sorted_frame_names[idx + shift] for shift in input_ids]
            hr_name = self.hr_suffix.join(input_frames[self.num_frames // 2].split(self.lr_suffix))
            if check_content and not check_file_existence(self.data_dir / hr_name):
                content_errors.append('{}: does not exist'.format(self.data_dir / hr_name))
            annotations.append(SuperResolutionAnnotation(
                MultiFramesInputIdentifier(input_ids, input_frames), hr_name, gt_loader=self.annotation_loader
            ))
            if progress_callback and idx % progress_interval == 0:
                progress_callback(idx * 100 / num_iterations)

        return ConverterReturn(annotations, None, content_errors)


class MultiTargetSuperResolutionConverter(BaseFormatConverter):
    __provider__ = 'multi_target_super_resolution'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'data_dir': PathField(description='Dataset root directory', is_directory=True),
            'lr_path': StringField(description='path to directory with low resolution images'),
            'hr_mapping': DictField(key_type=str, value_type=str, allow_empty=False),
            'annotation_loader': StringField(
                optional=True, choices=LOADERS_MAPPING.keys(), default='pillow',
                description="Which library will be used for ground truth image reading. "
                            "Supported: {}".format(', '.join(LOADERS_MAPPING.keys()))
            )
        })

        return params

    def configure(self):
        self.data_dir = self.get_value_from_config('data_dir')
        self.lr_dir = self.get_value_from_config('lr_path')
        self.hr_mapping = self.get_value_from_config('hr_mapping')
        self.annotation_loader = LOADERS_MAPPING.get(self.get_value_from_config('annotation_loader'))
        if not self.annotation_loader:
            raise ConfigError('provided not existing loader')
        self.full_lr_dir = Path(self.data_dir) / self.lr_dir
        if not self.full_lr_dir.exists():
            raise ConfigError('directory with low resolution images does not exist')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotations = []
        lr_images = list(self.full_lr_dir.glob('*lr.jpg'))
        for lr_image in lr_images:
            image_annotations = {}
            for hr_name, hr_dir in self.hr_mapping.items():
                hr_file = '{}/{}'.format(hr_dir, lr_image.name.replace('lr', 'hr'))
                image_annotations[hr_name] = SuperResolutionAnnotation(
                    '{}/{}'.format(self.lr_dir, lr_image.name), hr_file, gt_loader=self.annotation_loader
                )
            annotations.append(ContainerAnnotation(image_annotations))

        return ConverterReturn(annotations, None, None)
