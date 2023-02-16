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

from ..config import NumberField
from .preprocessor import Preprocessor


class PuTransferFunction(Preprocessor):
    """
    Fit of PU2 curve normalized at 100 cd/m^2
    [Aydin et al., 2008, "Extending Quality Metrics to Full Luminance Range Images"]
    """
    __provider__ = 'pu_transfer_function'

    def configure(self):
        self.params = {"pu_y0": 1.57945760e-06,
                       "pu_y1": 3.22087631e-02,
                       "pu_x0": 2.23151711e-03,
                       "pu_x1": 3.70974749e-01,
                       "pu_a": 1.41283765e+03,
                       "pu_b": 1.64593172e+00,
                       "pu_c": 4.31384981e-01,
                       "pu_d": -2.94139609e-03,
                       "pu_e": 1.92653254e-01,
                       "pu_f": 6.26026094e-03,
                       "pu_g": 9.98620152e-01}

        self.hdr_y_max = 65504.  # maximum HDR value
        self.pu_norm_scale = 1. / self.pu_forward(self.hdr_y_max, self.params)

    def process(self, image, annotation_meta=None):
        image.data[0] = self.pu_norm_scale * self.pu_forward(image.data[0], self.params)
        image.metadata['params'] = self.params
        return image

    @staticmethod
    def pu_forward(data, params):
        return np.where(data <= params["pu_y0"],
                        params["pu_a"] * data,
                        np.where(data <= params["pu_y1"],
                                 params["pu_b"] * np.power(data, params["pu_c"]) + params["pu_d"],
                                 params["pu_e"] * np.log(data + params["pu_f"]) + params["pu_g"]))
