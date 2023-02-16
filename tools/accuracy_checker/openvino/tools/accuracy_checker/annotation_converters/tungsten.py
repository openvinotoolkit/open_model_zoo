"""
Copyright (C) 2023 KNS Group LLC (YADRO)

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
from ..representation import ImageProcessingAnnotation
from ..config import PathField, ListField, StringField, ConfigError
from .format_converter import BaseFormatConverter, ConverterReturn
from ..utils import check_file_existence
from .image_processing import LOADERS_MAPPING


class TungstenAnnotationConverter(BaseFormatConverter):
    __provider__ = 'tungsten'
    annotation_types = (ImageProcessingAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'dataset_root_dir': PathField(is_directory=True, description="path to dataset root"),
            'extension': StringField(default='png', optional=True,
                                     description="images extension"),
            'features': ListField(value_type=str, default='color', optional=True,
                                  description='List of features'),
            'input_subfolder': ListField(value_type=str, default='spp_4_data', optional=True,
                                         description='sub-directory for input features'),
            'target_subfolder': StringField(
                optional=True,
                default='spp_4096_data',
                description="sub-directory for targets."
            ),
            'annotation_loader': StringField(
                optional=True, choices=LOADERS_MAPPING.keys(), default='opencv_unchanged',
                description="Which library will be used for ground truth image reading. "
                            "Supported: {}".format(', '.join(LOADERS_MAPPING.keys())))
        })

        return configuration_parameters

    def configure(self):
        self.dataset_root = self.get_value_from_config('dataset_root_dir')
        self.features = self.get_value_from_config('features')
        self.extension = self.get_value_from_config('extension')
        self.input_subfolder = self.get_value_from_config('input_subfolder')
        self.target_subfolder = self.get_value_from_config('target_subfolder')
        self.annotation_loader = LOADERS_MAPPING.get(self.get_value_from_config('annotation_loader'))
        if not self.annotation_loader:
            raise ConfigError('provided not existing loader')

    def convert(self, check_content=False, **kwargs):
        content_errors = None if not check_content else []
        annotations = []

        for scene in self.dataset_root.iterdir():
            scene_path = Path(scene)
            for folder in self.input_subfolder:
                path_data = scene_path / folder
                path_target = scene_path / self.target_subfolder
                num_images = len(list(Path(path_data).rglob(r'*_color.{}'.format(self.extension))))
                for idx in range(num_images):
                    color = path_data / f'{idx}_color.{self.extension}'
                    albedo = path_data / f'{idx}_albedo.{self.extension}'
                    target = path_target / f'{idx}_color.{self.extension}'
                    if check_content:
                        if not check_file_existence(color):
                            content_errors.append(f'{color}: does not exist')
                        if not check_file_existence(albedo):
                            content_errors.append(f'{albedo}: does not exist')
                    annotations.append(ImageProcessingAnnotation([str(color), str(albedo)], str(target),
                                                                 gt_loader=self.annotation_loader))

        return ConverterReturn(annotations, self.get_meta(), content_errors)
