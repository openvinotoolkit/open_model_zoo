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

import cv2
import numpy as np

from ..adapters import Adapter
from ..config import StringField
from ..representation import BackgroundMattingPrediction


class ImageBackgroundMattingAdapter(Adapter):
    __provider__ = 'background_matting_with_pha_and_fgr'

    def process(self, raw, identifiers, frame_meta):
        if not self.output_verified:
            self.select_output_blob(raw)
        result = []
        frame_meta = frame_meta or [] * len(identifiers)
        raw_outputs = self._extract_predictions(raw, frame_meta)
        pha = raw_outputs[self.pha]
        fgr = raw_outputs[self.fgr]
        batch_size = len(identifiers)
        for i in range(batch_size):
            output = {
                self._orig_pha: self.to_image(pha[i], frame_meta[i]),
                self._orig_fgr: self.to_image(fgr[i], frame_meta[i])
            }
            result.append(
                BackgroundMattingPrediction(identifiers[i], output)
            )
        return result

    @staticmethod
    def to_image(tensor, meta):
        out = cv2.resize(
            np.transpose(tensor, (1, 2, 0)),
            (meta['original_width'], meta['original_height'])
        )
        if len(out.shape) == 3 and out.shape[-1] == 3:
            out = out[:, :, ::-1]
        return out

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'alpha_out': StringField(
                description="Name of output layer with alpha.",
                optional=True
            ),
            'foreground_out': StringField(
                description="Name of output layer with foreground.",
                optional=True
            ),
        })

        return parameters

    def select_output_blob(self, outputs):
        self.pha = self.check_output_name(self.pha, outputs)
        self.fgr = self.check_output_name(self.fgr, outputs)
        self.output_verified = True

    def configure(self):
        self.pha = self.get_value_from_config('alpha_out')
        self._orig_pha = self.get_value_from_config('alpha_out')
        self.fgr = self.get_value_from_config('foreground_out')
        self._orig_fgr = self.get_value_from_config('foreground_out')
        self.output_verified = False
