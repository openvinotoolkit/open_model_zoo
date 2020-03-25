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

import cv2
import numpy as np
from ..config import PathField, StringField, BoolField, ConfigError, NumberField
from ..representation import SuperResolutionAnnotation
from ..representation.super_resolution_representation import GTLoader
from ..utils import check_file_existence
from ..data_readers import MultiFramesInputIdentifier
from .format_converter import BaseFormatConverter, ConverterReturn

LOADERS_MAPPING = {
    'opencv': GTLoader.OPENCV,
    'pillow': GTLoader.PILLOW
}


class SRConverter(BaseFormatConverter):
    __provider__ = 'super_resolution'
    annotation_types = (SuperResolutionAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'data_dir': PathField(
                is_directory=True, description="Path to folder, where images in low and high resolution are located."
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
        self.lr_suffix = self.get_value_from_config('lr_suffix')
        self.hr_suffix = self.get_value_from_config('hr_suffix')
        self.upsample_suffix = self.get_value_from_config('upsample_suffix')
        self.two_streams = self.get_value_from_config('two_streams')
        self.annotation_loader = LOADERS_MAPPING.get(self.get_value_from_config('annotation_loader'))
        if not self.annotation_loader:
            raise ConfigError('provided not existing loader')
        self.generate_upsample = self.get_value_from_config('generate_upsample')
        self.upsample_factor = self.get_value_from_config('upsample_factor')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        content_errors = [] if check_content else None
        file_list_lr = []
        for file_in_dir in self.data_dir.iterdir():
            if self.lr_suffix in file_in_dir.parts[-1]:
                file_list_lr.append(file_in_dir)

        annotation = []
        num_iterations = len(file_list_lr)
        for lr_id, lr_file in enumerate(file_list_lr):
            lr_file_name = lr_file.parts[-1]
            upsampled_file_name = self.upsample_suffix.join(lr_file_name.split(self.lr_suffix))
            if self.two_streams and self.generate_upsample:
                self.generate_upsample_file(lr_file, self.upsample_factor, upsampled_file_name)
            hr_file_name = self.hr_suffix.join(lr_file_name.split(self.lr_suffix))
            if check_content:
                if not check_file_existence(self.data_dir / hr_file_name):
                    content_errors.append('{}: does not exist'.format(self.data_dir / hr_file_name))
                if self.two_streams and not check_file_existence(self.data_dir / upsampled_file_name):
                    content_errors.append('{}: does not exist'.format(self.data_dir / upsampled_file_name))

            identifier = [lr_file_name, upsampled_file_name] if self.two_streams else lr_file_name
            annotation.append(SuperResolutionAnnotation(identifier, hr_file_name, gt_loader=self.annotation_loader))
            if progress_callback is not None and lr_id % progress_interval == 0:
                progress_callback(lr_id / num_iterations * 100)

        return ConverterReturn(annotation, None, content_errors)

    @staticmethod
    def generate_upsample_file(original_image_path, scale_factor, upsampled_file_name):
        image = cv2.imread(str(original_image_path))
        upsampled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(original_image_path.parent / upsampled_file_name), upsampled_image)


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
        content_errors = [] if check_content else None
        frames_ids = []
        frame_names = []
        annotations = []
        for file_in_dir in self.data_dir.iterdir():
            image_name = file_in_dir.parts[-1]
            if self.lr_suffix in image_name and self.hr_suffix not in image_name:
                frame_names.append(image_name)
                frames_ids.append(int(image_name.split(self.lr_suffix)[0]))
        sorted_frames = np.argsort(frames_ids)
        frames_ids.sort()
        sorted_frame_names = [frame_names[idx] for idx in sorted_frames]

        num_iterations = len(frames_ids)
        for idx, _ in enumerate(frames_ids):
            if len(frames_ids) - idx < self.num_frames:
                break
            input_ids = list(range(self.num_frames))
            input_frames = [sorted_frame_names[idx + shift] for shift in input_ids]
            hr_name = self.hr_suffix.join(input_frames[0].split(self.lr_suffix))
            if check_content and not check_file_existence(self.data_dir / hr_name):
                content_errors.append('{}: does not exist'.format(self.data_dir / hr_name))
            annotations.append(SuperResolutionAnnotation(
                MultiFramesInputIdentifier(input_ids, input_frames), hr_name, gt_loader=self.annotation_loader
            ))
            if progress_callback and idx % progress_interval == 0:
                progress_callback(idx * 100 / num_iterations)

        return ConverterReturn(annotations, None, content_errors)
