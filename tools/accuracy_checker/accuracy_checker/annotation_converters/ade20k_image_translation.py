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

from ..config import PathField
from ..representation import ImageProcessingAnnotation
from .format_converter import BaseFormatConverter, ConverterReturn


class ADE20kImageTranslationConverter(BaseFormatConverter):
    __provider__ = 'ade20k_image_translation'
    annotation_types = (ImageProcessingAnnotation,)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'annotations_dir': PathField(
                optional=False, is_directory=True,
                description="Path to directory with masks."
            ),
            'reference_file': PathField(
                optional=False, description="Path to .txt file with pairs (validation:train)."
            )
        })
        return parameters

    def configure(self):
        self.annotation_dir = self.get_value_from_config('annotations_dir')
        self.reference_dict = self.get_value_from_config('reference_file')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotations = []
        ref_dict = self.get_ref()
        for key, value in ref_dict.items():
            input_mask_filename = "annotations/validation/" + key.replace('.jpg', '.png')
            reference_image_filename = "images/training/" + value
            reference_mask_filename = "annotations/training/" + value.replace('.jpg', '.png')
            identifier = [str(input_mask_filename), str(reference_image_filename), str(reference_mask_filename)]
            annotation = ImageProcessingAnnotation(identifier, "validation/{}".format(key))
            annotations.append(annotation)

        return ConverterReturn(annotations, None, None)

    def get_ref(self):
        with open(self.reference_dict, 'r', encoding="utf-8-sig") as fd:
            lines = fd.readlines()
        ref_dict = {}
        for line in lines:
            key, value = line.strip().split(',')[:2]
            ref_dict[key] = value
        return ref_dict
