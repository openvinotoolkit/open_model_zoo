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

from .base_representation import BaseRepresentation

class Text2ImageGeneration(BaseRepresentation):
    pass


class Text2ImageGenerationAnnotation(Text2ImageGeneration):
    def __init__(self, identifiers, prompt):
        super().__init__(identifiers)
        self.prompt = prompt


class Text2ImageGenerationPrediction(Text2ImageGeneration):
    def __init__(self, identifiers, prediction):
        """
        Args:
            identifiers: object identifier (e.g. image name).
            prediction: array with shape (height, width) contained result image.
        """

        super().__init__(identifiers)
        self.value = prediction
