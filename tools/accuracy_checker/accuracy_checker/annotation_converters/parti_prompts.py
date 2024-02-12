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

from ..config import PathField
from ..representation import Text2ImageGenerationAnnotation
from ..utils import read_csv
from .format_converter import BaseFormatConverter, ConverterReturn


class PartiPromptsDatasetConverter(BaseFormatConverter):
    __provider__ = "parti_prompts"

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'annotation_file': PathField(description='path to annotation file in csv format'),
        })

        return params

    def configure(self):
        self.annotation_file = self.get_value_from_config("annotation_file")

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        data = read_csv(self.annotation_file, delimiter="\t")
        num_iter = len(data)
        annotations = []
        for example_id, line in enumerate(data):
            annotations.append(Text2ImageGenerationAnnotation(example_id, line["Prompt"]))
            if progress_callback and example_id % progress_interval == 0:
                progress_callback(example_id * 100 / num_iter)

        return ConverterReturn(annotations, {}, None)
