"""
Copyright (c) 2024 Intel Corporation

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

import numpy as np
from .base_representation import BaseRepresentation

class ImageFeatureAnnotation(BaseRepresentation):
    def __init__(self, identifier, sequence):
        super().__init__(identifier)
        self.sequence = sequence

class ImageFeaturePrediction(BaseRepresentation):
    def __init__(self, identifier, matches0, matching_scores0):
        super().__init__(identifier)
        self.matches0 = matches0 if matches0 is not None else np.array([])
        self.matching_scores0 = matching_scores0 if matching_scores0 is not None else np.array([])
