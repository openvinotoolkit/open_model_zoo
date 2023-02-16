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
import numpy as np

from .postprocessor import Postprocessor
from ..config import NumberField
from ..representation import ImageProcessingPrediction, ImageProcessingAnnotation
from ..preprocessor import PuTransferFunction


class PuInverseTransferFunction(Postprocessor):
    """
        Fit of PU2 curve normalized at 100 cd/m^2
        [Aydin et al., 2008, "Extending Quality Metrics to Full Luminance Range Images"]
    """
    __provider__ = 'pu_inverse_transfer_function'

    prediction_types = (ImageProcessingAnnotation, )
    annotation_types = (ImageProcessingPrediction, )

    def configure(self):
        self.hdr_y_max = 65504.  # maximum HDR value

    def process_image_with_metadata(self, annotation, prediction, image_metadata=None):
        for prediction_, annotation_ in zip(prediction, annotation):
            params = image_metadata.get('params', None)
            pu_norm_scale = 1. / PuTransferFunction.pu_forward(self.hdr_y_max, params)
            prediction_.value = self.pu_inverse(prediction_.value / pu_norm_scale, params)

        return annotation, prediction

    @staticmethod
    def pu_inverse(data, params):
        return np.where(data <= params["pu_x0"],
                        data / params["pu_a"],
                        np.where(data <= params["pu_x1"],
                                 np.power((data - params["pu_d"]) / params["pu_b"], 1. / params["pu_c"]),
                                 np.exp((data - params["pu_g"]) / params["pu_e"]) - params["pu_f"]))
