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
import warnings
from pathlib import Path
import cv2
import numpy as np
from ..config import PathField, StringField, BoolField, ConfigError, NumberField, DictField, BaseField
from ..representation import SuperResolutionAnnotation, ContainerAnnotation
from ..representation.image_processing import GTLoader
from ..utils import check_file_existence, get_path
from ..data_readers import MultiFramesInputIdentifier
from .format_converter import BaseFormatConverter, ConverterReturn

LOADERS_MAPPING = {
    'opencv': GTLoader.OPENCV,
    'pillow': GTLoader.PILLOW,
    'dicom': GTLoader.DICOM,
    'skimage': GTLoader.SKIMAGE,
    'pillow_rgb': GTLoader.PILLOW_RGB
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
            'images_dir': PathField(
                optional=True, is_directory=True,
                description="Path to directory with images.",
            ),
            'lr_suffix': StringField(
                optional=True, default="lr", description="Low resolution file name's suffix."
            ),
            'hr_suffix': StringField(
                optional=True, default="hr", description="High resolution file name's suffix."
            ),
            'ignore_suffixes': BoolField(optional=True, default=False),
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
        self.data_dir = self.get_value_from_config('data_dir') or self.get_value_from_config('images_dir')
        self.lr_dir = self.get_value_from_config('lr_dir') or self.data_dir
        if not self.lr_dir:
            raise ConfigError('One of the parameters: data_dir or lr_dir should be provided for conversion')
        self.hr_dir = self.get_value_from_config('hr_dir') or self.data_dir
        self.lr_suffix = self.get_value_from_config('lr_suffix')
        self.hr_suffix = self.get_value_from_config('hr_suffix')
        self.upsample_suffix = self.get_value_from_config('upsample_suffix')
        self.two_streams = self.get_value_from_config('two_streams')
        self.ignore_suffixes = self.get_value_from_config('ignore_suffixes')
        self.relative_dir = ''
        if self.two_streams:
            self.upsampled_dir = self.get_value_from_config('upsampled_dir')
            if not self.upsampled_dir:
                self.upsampled_dir = self.lr_dir

            if self.data_dir:
                try:
                    self.lr_dir.relative_to(self.data_dir)
                    self.upsampled_dir.relative_to(self.data_dir)
                except ValueError:
                    raise ConfigError('data_dir parameter should be provided for conversion as common part of paths '
                                      'lr_dir and upsampled_dir, if 2 streams used')
            self.relative_dir = self.data_dir or os.path.commonpath([self.lr_dir, self.upsampled_dir])
            if self.lr_dir != self.upsampled_dir:
                warnings.warn("lr_dir and upsampled_dir are different folders."
                              "Make sure that data_source is {}".format(self.relative_dir))

        self.annotation_loader = LOADERS_MAPPING.get(self.get_value_from_config('annotation_loader'))
        if not self.annotation_loader:
            raise ConfigError('provided not existing loader')
        self.generate_upsample = self.get_value_from_config('generate_upsample')
        self.upsample_factor = self.get_value_from_config('upsample_factor')
        if self.ignore_suffixes:
            if self.hr_dir is None:
                raise ConfigError('please provide hr_dir')
            if self.lr_dir == self.hr_dir:
                raise ConfigError(
                    'high and low resolution images should be located in separated directories '
                    'if ignore_suffixes enabled')
            if self.two_streams and self.lr_dir == self.upsampled_dir:
                raise ConfigError(
                    'low resolution and upsample images should be located in separated directories '
                    'if ignore_suffixes enabled')
            if self.two_streams and self.hr_dir == self.upsampled_dir:
                raise ConfigError(
                    'high resolution and upsample images should be located in separated directories '
                    'if ignore_suffixes enabled')


    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        content_errors = [] if check_content else None
        file_list_lr = self.get_lr_list()
        annotation = []
        num_iterations = len(file_list_lr)
        for lr_id, lr_file in enumerate(file_list_lr):
            lr_file_name = lr_file.parts[-1]
            upsampled_file_name = ''
            if self.two_streams:
                if self.generate_upsample:
                    upsampled_file_name = self.upsample_suffix.join(
                        lr_file_name.split(self.lr_suffix) if self.lr_suffix else [lr_file_name, '']
                    ) if not self.ignore_suffixes else lr_file_name
                    self.generate_upsample_file(lr_file, self.upsample_factor, upsampled_file_name, self.upsampled_dir)
                else:
                    if self.ignore_suffixes:
                        idx = self.get_index(lr_file_name, '')
                        ups_files = list(self.upsampled_dir.glob('*{}*'.format(idx)))
                        if not ups_files:
                            continue
                        upsampled_file_name = ups_files[0].name
                    else:
                        upsampled_file_name = self.upsample_suffix.join(
                            lr_file_name.split(self.lr_suffix) if self.lr_suffix else [lr_file_name, '']
                        )

            if not self.ignore_suffixes:
                hr_file_name = self.hr_suffix.join(
                    lr_file_name.split(self.lr_suffix) if self.lr_suffix else [lr_file_name, '']
                )
            else:
                idx = self.get_index(lr_file_name, '')
                hr_files = list(self.hr_dir.glob('*{}*'.format(idx)))
                if not hr_files:
                    continue
                hr_file_name = hr_files[0].name
            if check_content:
                content_errors.extend(self.check_content(hr_file_name, upsampled_file_name))

            identifier = self.generate_identifier(lr_file_name, lr_file, upsampled_file_name)
            annotation.append(SuperResolutionAnnotation(identifier, hr_file_name, gt_loader=self.annotation_loader))
            if progress_callback is not None and lr_id % progress_interval == 0:
                progress_callback(lr_id / num_iterations * 100)

        return ConverterReturn(annotation, None, content_errors)

    def generate_identifier(self, lr_file_name, lr_file, upsampled_file_name):
        identifier = lr_file_name
        if self.two_streams:
            identifier = [str(lr_file.relative_to(self.relative_dir)),
                          str(Path(self.upsampled_dir, upsampled_file_name).relative_to(self.relative_dir))]
        return identifier

    @staticmethod
    def generate_upsample_file(original_image_path, scale_factor, upsampled_file_name, upsampled_image_path):
        image = cv2.imread(str(original_image_path))
        upsampled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(upsampled_image_path / upsampled_file_name), upsampled_image)

    @staticmethod
    def get_index(image_name, suffix):
        name_parts = image_name.split(suffix) if suffix else [image_name]
        numbers = []
        for part in name_parts:
            numbers += [int(s) for s in re.findall(r'\d+', part)]
        if not numbers:
            raise ValueError('no numeric in {}'.format(image_name))
        return numbers[0]

    def get_lr_list(self):
        file_list_lr = []
        for file_in_dir in self.lr_dir.iterdir():
            if self.lr_suffix in file_in_dir.parts[-1] or self.ignore_suffixes:
                file_list_lr.append(file_in_dir)
        return file_list_lr

    def check_content(self, hr_file_name, upsampled_file_name):
        content_errors = []
        if self.hr_dir and not check_file_existence(self.hr_dir / hr_file_name):
            content_errors.append('{}: does not exist'.format(self.hr_dir / hr_file_name))
        if self.two_streams and not check_file_existence(self.upsampled_dir / upsampled_file_name):
            content_errors.append('{}: does not exist'.format(self.upsampled_dir / upsampled_file_name))
        return content_errors


class SRMultiFrameConverter(BaseFormatConverter):
    __provider__ = 'multi_frame_super_resolution'
    annotation_types = (SuperResolutionAnnotation, )
    predefined_ref_frame = {
        'first': lambda x: 0,
        'middle': lambda x: int(x) // 2,
        'last': lambda x: -1
    }

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
            ),
            'reference_frame': BaseField(optional=True, default='middle', description='id of frame used as reference')
        })
        return params

    def configure(self):
        self.data_dir = self.get_value_from_config('data_dir')
        self.lr_suffix = self.get_value_from_config('lr_suffix')
        self.hr_suffix = self.get_value_from_config('hr_suffix')
        self.annotation_loader = LOADERS_MAPPING.get(self.get_value_from_config('annotation_loader'))
        self.num_frames = self.get_value_from_config('number_input_frames')
        self.max_frame_id = self.get_value_from_config('max_frame_id')
        self.reference_frame = self.parse_ref_frame(self.get_value_from_config('reference_frame'), self.num_frames)

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
            hr_name = self.hr_suffix.join(input_frames[self.reference_frame].split(self.lr_suffix))
            if check_content and not check_file_existence(self.data_dir / hr_name):
                content_errors.append('{}: does not exist'.format(self.data_dir / hr_name))
            annotations.append(SuperResolutionAnnotation(
                MultiFramesInputIdentifier(input_ids, input_frames), hr_name, gt_loader=self.annotation_loader
            ))
            if progress_callback and idx % progress_interval == 0:
                progress_callback(idx * 100 / num_iterations)

        return ConverterReturn(annotations, None, content_errors)

    def parse_ref_frame(self, config_value, num_frames):
        try:
            ref_frame = int(config_value)
        except ValueError:
            ref_func = self.predefined_ref_frame.get(config_value)
            if ref_func is None:
                raise ConfigError('Unsupported value for reference_frame: {}'.format(config_value))
            ref_frame = ref_func(num_frames)
        if ref_frame > num_frames:
            raise ConfigError('Unexpected value for reference_frame id: {}'.format(ref_frame))
        return ref_frame


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


class SRDirectoryBased(BaseFormatConverter):
    __provider__ = 'super_resolution_dir_based'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'images_dir': PathField(is_directory=True, description='dataset roo dir', optional=True),
            'lr_dir': PathField(optional=True, description='directory with low resolution images', is_directory=True),
            'hr_dir': PathField(optional=True, description='directory with high resolution images', is_directory=True),
            'upsampled_dir': PathField(optional=True, description='directory with upsampled images', is_directory=True),
            'two_streams': BoolField(optional=True, default=False),
            'hr_prefixed': BoolField(optional=True, default=False),
            'annotation_loader': StringField(
                optional=True, choices=LOADERS_MAPPING.keys(), default='pillow',
                description="Which library will be used for ground truth image reading. "
                            "Supported: {}".format(', '.join(LOADERS_MAPPING.keys()))
            ),
            'relaxed_names': BoolField(optional=True, default=False)
        })
        return parameters

    def configure(self):
        def set_default_path(add_dir, param_name):
            if self.images_dir is None:
                raise ConfigError(error_msg_not_provided.format(param_name, 'images_dir'))
            setattr(self, param_name, get_path(self.images_dir / add_dir, is_directory=True))

        self.images_dir = self.get_value_from_config('images_dir')
        self.annotation_loader = LOADERS_MAPPING.get(self.get_value_from_config('annotation_loader'))
        self.relaxed_names = self.get_value_from_config('relaxed_names')
        self.lr_dir = self.get_value_from_config('lr_dir')
        self.hr_dir = self.get_value_from_config('hr_dir')
        self.upsample_dir = self.get_value_from_config('upsampled_dir')
        self.two_streams = self.get_value_from_config('two_streams')
        self.hr_prefixed = self.get_value_from_config('hr_prefixed')
        error_msg_not_provided = '{} or {} should be provided'
        error_msg_the_same_dir = '{} and {} should contain different directories'
        if self.lr_dir is None:
            set_default_path('LR', 'lr_dir')
        if self.hr_dir is None:
            set_default_path('HR', 'hr_dir')
        if self.two_streams and self.upsample_dir is None:
            set_default_path('upsample', 'upsample_dir')
        if self.lr_dir == self.hr_dir:
            raise ConfigError(error_msg_the_same_dir.format('lr_dir', 'hr_dit'))
        if self.two_streams:
            if self.lr_dir == self.upsample_dir or self.hr_dir == self.upsample_dir:
                raise ConfigError(error_msg_the_same_dir.format('upsample_dir, hr_dir', 'lr_dir'))
        if self.images_dir:
            try:
                self.lr_dir.relative_to(self.images_dir)
            except ValueError:
                raise ConfigError('lr_dir should be relative to images_dir')
            if self.two_streams:
                try:
                    self.upsample_dir.relative_to(self.images_dir)
                except ValueError:
                    raise ConfigError('upsample_dir should be relative to images_dir')
        else:
            self.images_dir = (
                os.path.commonpath([str(self.lr_dir), str(self.upsample_dir)]) if self.two_streams else self.lr_dir
            )

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        content_errors = [] if check_content else None
        file_list_lr = list(self.lr_dir.iterdir())
        annotation = []
        num_iterations = len(file_list_lr)
        for lr_id, lr_file in enumerate(file_list_lr):
            lr_file_name = lr_file.name
            hr_file = (
                self.hr_dir / lr_file_name if not self.relaxed_names and not self.hr_prefixed
                else self.find_file_by_id(self.hr_dir, lr_file_name) if self.relaxed_names
                else self.find_file_with_prefix(self.hr_dir, lr_file_name)
            )
            if hr_file is None:
                continue
            if check_content and not check_file_existence(hr_file):
                content_errors.append("{}: does not exist".format(hr_file))
            if not self.two_streams:
                annotation.append(
                    SuperResolutionAnnotation(
                        str(lr_file.relative_to(self.images_dir)), hr_file.name, gt_loader=self.annotation_loader)
                )
            else:
                upsample_file = (
                    self.upsample_dir / lr_file_name if not self.relaxed_names
                    else self.find_file_by_id(self.upsample_dir, lr_file_name)
                )
                if upsample_file is None:
                    continue
                if check_content and not check_file_existence(upsample_file):
                    content_errors.append("{}: does not exist".format(upsample_file))
                annotation.append(
                    SuperResolutionAnnotation(
                        [str(lr_file.relative_to(self.images_dir)), str(upsample_file.relative_to(self.images_dir))],
                        hr_file.name, gt_loader=self.annotation_loader
                    )
                )
            if progress_callback and lr_id % progress_interval == 0:
                progress_callback(lr_id * 100 / num_iterations)

        return ConverterReturn(annotation, None, content_errors)

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
