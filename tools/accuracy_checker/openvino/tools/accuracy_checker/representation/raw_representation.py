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

from .base_representation import BaseRepresentation


class RawTensorRepresentation(BaseRepresentation):
    pass


class RawTensorAnnotation(RawTensorRepresentation):
    def __init__(self, identifier, annotation):
        super().__init__(identifier)
        self.value = annotation


class RawTensorPrediction(RawTensorRepresentation):
    def __init__(self, identifiers, prediction):
        super().__init__(identifiers)
        self.value = prediction
