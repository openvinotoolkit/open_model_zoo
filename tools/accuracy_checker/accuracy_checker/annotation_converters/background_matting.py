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
                'foregrounds_dir': PathField(optional=True,
                                             description='path to input foregrounds directory', is_directory=True),
                'foreground_prefix': StringField(optional=True, default='', description='prefix for gt foregrounds'),
                'foreground_postfix': StringField(optional=True, default='.png', description='prefix for foregrounds'),
                'with_foreground': BoolField(optional=True, default=False, description='load foregrounds'),
                'with_alpha': BoolField(optional=True, default=False,
                                        description='load images with mask including alpha channel'),
                "per_clip_location": BoolField(optional=True, default=False, description="inverse data struction")
            }
        )
        return configuration_parameters

    def configure(self):
        super().configure()
        self.backgrounds_dir = self.get_value_from_config('backgrounds_dir')
        self.background_prefix = self.get_value_from_config('background_prefix')
        self.background_postfix = self.get_value_from_config('background_postfix')
        self.with_background = self.get_value_from_config('with_background')
        self.foregrounds_dir = self.get_value_from_config('foregrounds_dir')
        self.foreground_prefix = self.get_value_from_config('foreground_prefix')
        self.foreground_postfix = self.get_value_from_config('foreground_postfix')
        self.with_foreground = self.get_value_from_config('with_foreground')
        self.with_alpha = self.get_value_from_config('with_alpha')
        self.per_clip_location = self.get_value_from_config("per_clip_location")

    # pylint: disable=too-many-branches
    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotations = []
        if not self.per_clip_location:
            image_name = '{prefix}{clip}/{base}{postfix}'.format(
                prefix=self.images_prefix, clip='{clip}', base='{base}', postfix=self.images_postfix
            )
            mask_name = '{prefix}{clip}/{base}{postfix}'.format(
                prefix=self.mask_prefix, clip='{clip}', base='{base}', postfix=self.mask_postfix
            )
        else:
            image_name = "{clip}/{prefix}/{base}{postfix}".format(
                prefix=self.images_prefix, clip='{clip}', base='{base}', postfix=self.images_postfix
            )
            mask_name = '{clip}/{prefix}/{base}{postfix}'.format(
                prefix=self.mask_prefix, clip='{clip}', base='{base}', postfix=self.mask_postfix
            )

        image_pattern = self.get_image_pattern('**/*')
        images_list = list(self.images_dir.glob(image_pattern))
        clips_list = self.get_clips_names(images_list, self.per_clip_location)
        num_iterations = len(images_list)
        content_errors = None if not check_content else []
        idx = 0
        for clip_name in sorted(clips_list):
            clip_dir = (self.images_dir / self.images_prefix / clip_name
                        if not self.per_clip_location else self.images_dir / clip_name / self.images_prefix
                        )
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
                    if not self.per_clip_location:
                        bgr_name = '{prefix}{clip}/{base}{postfix}'.format(
                            prefix=self.background_prefix,
                            clip=clip_name,
                            base=base_name,
                            postfix=self.background_postfix
                        )
                    else:
                        bgr_name = '{clip}/{prefix}/{base}{postfix}'.format(
                            prefix=self.background_prefix,
                            clip=clip_name,
                            base=base_name,
                            postfix=self.background_postfix
                        )
                    bgr_file = self.backgrounds_dir / bgr_name
                    if not bgr_file.exists():
                        continue
                    identifier = [identifier, bgr_name]

                fgr_name = None
                if self.with_foreground:
                    if not self.per_clip_location:
                        fgr_name = '{prefix}{clip}/{base}{postfix}'.format(
                            prefix=self.foreground_prefix,
                            clip=clip_name,
                            base=base_name,
                            postfix=self.foreground_postfix
                        )
                    else:
                        fgr_name = '{clip}/{prefix}/{base}{postfix}'.format(
                            prefix=self.foreground_prefix,
                            clip=clip_name,
                            base=base_name,
                            postfix=self.foreground_postfix
                        )
                    fgr_file = self.foregrounds_dir / fgr_name
                    if not fgr_file.exists():
                        continue

                annotations.append(
                    BackgroundMattingAnnotation(identifier, mask, self.mask_to_gray,
                                                self.with_alpha, path_to_fgr=fgr_name, video_id=clip_name)
                )
                if progress_callback is not None and idx % progress_interval == 0:
                    progress_callback(idx / num_iterations * 100)
                idx += 1

        return ConverterReturn(
            annotations, self.get_meta(), content_errors
        )

    @staticmethod
    def get_clips_names(images_list, per_clip_location):
        result = []
        for image in images_list:
            clip_path = image.parent if not per_clip_location else image.parents[1]
            result.append(clip_path.name)
        return list(set(result))

    def get_image_pattern(self, image_pattern='*', with_prefix=True, with_postfix=True):
        if with_prefix and self.images_prefix:
            image_pattern = self.images_prefix + image_pattern
            if self.per_clip_location:
                image_pattern = "*/" + image_pattern
        if with_postfix and self.images_postfix:
            image_pattern = image_pattern + self.images_postfix
        return image_pattern
