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

import itertools
from .postprocessor import Postprocessor
from ..representation import MachineTranslationAnnotation, MachineTranslationPrediction


class RemoveRepeatTokens(Postprocessor):
    __provider__ = 'remove_repeats'
    annotation_types = (MachineTranslationAnnotation, )
    prediction_types = (MachineTranslationPrediction, )

    def process_image(self, annotation, prediction, image_metadata=None):
        for prediction_ in prediction:
            tokens = prediction_.translation
            prediction_.translation = [token for token, _ in itertools.groupby(tokens)]
        return annotation, prediction
