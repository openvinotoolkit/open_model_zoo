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

from ..config import NumberField
from .postprocessor import Postprocessor
from ..representation import ImageProcessingPrediction, ImageProcessingAnnotation
from ..preprocessor import AutoExposure


class AutoExposureImage(Postprocessor):
    __provider__ = 'autoexposure'

    prediction_types = (ImageProcessingAnnotation, )
    annotation_types = (ImageProcessingPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'key': NumberField(value_type=float, optional=True, description="Destination width."),
            'k': NumberField(value_type=int, optional=True, min_value=1, description="Downsampling amount"),
        })
        return parameters

    def configure(self):
        self.params = {'key': self.get_value_from_config('key'),
                       'k': self.get_value_from_config('k')}

    def process_image(self, annotation, prediction):
        for prediction_, annotation_ in zip(prediction, annotation):
            exposure = annotation_.metadata.get('exposure', None)
            prediction_.value = prediction_.value / exposure if exposure \
                else prediction_.value / AutoExposure.autoexposure(prediction_.value, self.params)

        return annotation, prediction
