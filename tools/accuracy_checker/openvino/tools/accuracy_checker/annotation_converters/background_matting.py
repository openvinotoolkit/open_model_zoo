"""
Copyright (c) 2018-2022 Intel Corporation

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

from ..config import PathField, StringField, BoolField
from .format_converter import BaseFormatConverter, ConverterReturn
from ..representation import BackgroundMattingAnnotation
from ..data_readers import VideoFrameIdentifier


class BackgroundMattingConverter(BaseFormatConverter):
    __provider__ = 'background_matting'

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update(
            {
                'images_dir': PathField(description='path to input images directory', is_directory=True),
                'masks_dir': PathField(description='path to gt masks directory', is_directory=True),
                'image_prefix': StringField(optional=True, default='', description='prefix for images'),
                'mask_prefix': StringField(optional=True, default='', description='prefix for gt masks'),
                'image_postfix': StringField(optional=True, default='.png', description='prefix for images'),
                'mask_postfix': StringField(optional=True, default='.png', description='prefix for gt masks'),
                'mask_to_gray': BoolField(
                    optional=True, default=False, description='allow converting mask to grayscale'
                )
            }
        )
        return configuration_parameters

    def configure(self):
        self.images_dir = self.get_value_from_config('images_dir')
        self.masks_dir = self.get_value_from_config('masks_dir')
        self.images_prefix = self.get_value_from_config('image_prefix')
        self.images_postfix = self.get_value_from_config('image_postfix')
        self.mask_prefix = self.get_value_from_config('mask_prefix')
        self.mask_postfix = self.get_value_from_config('mask_postfix')
        self.dataset_meta = self.get_value_from_config('dataset_meta_file')
        self.mask_to_gray = self.get_value_from_config('mask_to_gray')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotations = []
        mask_name = '{prefix}{base}{postfix}'.format(
            prefix=self.mask_prefix, base='{base}', postfix=self.mask_postfix
        )
        image_pattern = '*'
        if self.images_prefix:
            image_pattern = self.images_prefix + image_pattern
        if self.images_postfix:
            image_pattern = image_pattern + self.images_postfix
        images_list = list(self.images_dir.glob(image_pattern))
        num_iterations = len(images_list)
        content_errors = None if not check_content else []
        for idx, image in enumerate(images_list):
            base_name = image.name
            identifier = base_name
            if self.images_prefix:
                base_name = base_name.split(self.images_prefix)[-1]
            if self.images_postfix:
                base_name = base_name.split(self.images_postfix)[0]

            mask_file = self.masks_dir / mask_name.format(base=base_name)
            if not mask_file.exists():
                continue

            annotations.append(
                BackgroundMattingAnnotation(identifier, mask_file.name, self.mask_to_gray)
            )
            if progress_callback is not None and idx % progress_interval == 0:
                progress_callback(idx / num_iterations * 100)

        return ConverterReturn(
            annotations, self.get_meta(), content_errors
        )

    def get_meta(self):
        return {'label_map': {'background': 0, 'foreground': list(range(1, 256))}}


class VideoBackgroundMatting(BackgroundMattingConverter):
    __provider__ = 'video_background_matting'

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotations = []
        mask_name = '{prefix}{base}{postfix}'.format(
            prefix=self.mask_prefix, base='{base}', postfix=self.mask_postfix
        )
        image_pattern = '*'
        if self.images_prefix:
            image_pattern = self.images_prefix + image_pattern
        if self.images_postfix:
            image_pattern = image_pattern + self.images_postfix
        images_list = sorted(self.images_dir.glob(image_pattern))
        num_iterations = len(images_list)
        content_errors = None if not check_content else []
        for idx, image in enumerate(images_list):
            base_name = image.name
            if '.mp4' not in base_name:
                continue
            video_id = base_name.split('.mp4')[0]
            identifier = VideoFrameIdentifier(video_id, base_name)
            if self.images_prefix:
                base_name = base_name.split(self.images_prefix)[-1]
            if self.images_postfix:
                base_name = base_name.split(self.images_postfix)[0]

            mask_file = self.masks_dir / mask_name.format(base=base_name)
            if not mask_file.exists():
                continue

            annotations.append(
                BackgroundMattingAnnotation(identifier, mask_file.name, self.mask_to_gray)
            )
            if progress_callback is not None and idx % progress_interval == 0:
                progress_callback(idx / num_iterations * 100)

        return ConverterReturn(
            annotations, self.get_meta(), content_errors
        )


class BackgroundMattingSequential(BackgroundMattingConverter):
    __provider__ = 'background_matting_sequential'

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update(
            {
                'backgrounds_dir': PathField(description='path to input backgrounds directory', is_directory=True),
                'background_prefix': StringField(optional=True, default='', description='prefix for gt backgrounds'),
                'background_postfix': StringField(optional=True, default='.png', description='prefix for backgrounds'),
                'with_background': BoolField(optional=True, default=False, description='load backgrounds'),
                'with_alpha': BoolField(optional=True, default=False,
                                        description='load images with mask including alpha channel'),
            }
        )
        return configuration_parameters

    def configure(self):
        super().configure()
        self.backgrounds_dir = self.get_value_from_config('backgrounds_dir')
        self.background_prefix = self.get_value_from_config('background_prefix')
        self.background_postfix = self.get_value_from_config('background_postfix')
        self.with_background = self.get_value_from_config('with_background')
        self.with_alpha = self.get_value_from_config('with_alpha')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotations = []
        image_name = '{prefix}{clip}/{base}{postfix}'.format(
            prefix=self.images_prefix, clip='{clip}', base='{base}', postfix=self.images_postfix
        )
        mask_name = '{prefix}{clip}/{base}{postfix}'.format(
            prefix=self.mask_prefix, clip='{clip}', base='{base}', postfix=self.mask_postfix
        )
        image_pattern = self.get_image_pattern('**/*')
        images_list = list(self.images_dir.glob(image_pattern))
        clips_list = self.get_clips_names(images_list)
        num_iterations = len(images_list)
        content_errors = None if not check_content else []
        idx = 0
        for clip_name in sorted(clips_list):
            clip_dir = self.images_dir / self.images_prefix / clip_name
            image_pattern = self.get_image_pattern('*', with_prefix=False)
            clip_images = list(clip_dir.glob(image_pattern))

            for image in sorted(clip_images):
                base_name = image.name
                if self.images_prefix:
                    base_name = base_name.split(self.images_prefix)[-1]
                if self.images_postfix:
                    base_name = base_name.split(self.images_postfix)[0]

                mask_file = self.masks_dir / mask_name.format(base=base_name, clip=clip_name)
                image_file = self.images_dir / image_name.format(base=base_name, clip=clip_name)
                if not (mask_file.exists() and image_file.exists()):
                    continue
                identifier = image_name.format(base=base_name, clip=clip_name)
                mask = mask_name.format(base=base_name, clip=clip_name)

                if self.with_background:
                    bgr_name = '{prefix}{clip}/{base}{postfix}'.format(
                        prefix=self.background_prefix, clip=clip_name, base=base_name, postfix=self.background_postfix
                    )
                    bgr_file = self.backgrounds_dir / bgr_name
                    if not bgr_file.exists():
                        continue
                    identifier = [identifier, bgr_name]

                annotations.append(
                    BackgroundMattingAnnotation(identifier, mask, self.mask_to_gray,
                                                self.with_alpha, video_id=clip_name)
                )
                if progress_callback is not None and idx % progress_interval == 0:
                    progress_callback(idx / num_iterations * 100)
                idx += 1

        return ConverterReturn(
            annotations, self.get_meta(), content_errors
        )

    @staticmethod
    def get_clips_names(images_list):
        result = []
        for image in images_list:
            clip_path = image.parent
            result.append(clip_path.name)
        return list(set(result))

    def get_image_pattern(self, image_pattern='*', with_prefix=True, with_postfix=True):
        if with_prefix and self.images_prefix:
            image_pattern = self.images_prefix + image_pattern
        if with_postfix and self.images_postfix:
            image_pattern = image_pattern + self.images_postfix
        return image_pattern
