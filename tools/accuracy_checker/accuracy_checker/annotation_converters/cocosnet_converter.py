"""
Copyright (c) 2020 Intel Corporation

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
from ..representation import CocosnetAnnotation, CocosnetPrediction
from .format_converter import BaseFormatConverter, ConverterReturn


class CocosnetConverter(BaseFormatConverter):
    __provider__ = 'cocosnet'
    annotation_types = (CocosnetAnnotation,)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'images_dir': PathField(
                optional=False, is_directory=True,
                description="Path to directory with images."
            ),
            'annotations_dir': PathField(
                optional=False, is_directory=True,
                description="Path to directory with masks."
            ),
            'reference_dict': PathField(
                optional=False, description="Path to .txt file with pairs (validation:train)."
            )
        })
        return parameters

    
    def configure(self):
        self.image_dir = self.get_value_from_config('images_dir')
        self.annotation_dir = self.get_value_from_config('annotations_dir')
        self.reference_dict = self.get_value_from_config('reference_dict')

    
    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        content_check_errors = [] if check_content else None
        annotations = []
        ref_dict = self.get_ref()
        images = list(im for im in self.image_dir.iterdir())
        for key, value in ref_dict.items():
            input_mask_filename = self.annotation_dir / "validation" /  key.replace('.jpg', '.png')
            reference_image_filename = self.image_dir / "training" / value
            reference_mask_filename = self.annotation_dir / "training" / value.replace('.jpg', '.png')
            identifier = [str(input_mask_filename), str(reference_image_filename), str(reference_mask_filename)]
            annotation = CocosnetAnnotation(identifier, self.image_dir / "validation" / key)
            annotations.append(annotation)

        return ConverterReturn(annotations, None, content_check_errors)

    
    def get_ref(self):
        with open(self.reference_dict) as fd:
            lines = fd.readlines()
        ref_dict = {}
        for i in range(len(lines)):
            items = lines[i].strip().split(',')
            key = items[0]
            value = items[1]
            ref_dict[key] = value
        return ref_dict