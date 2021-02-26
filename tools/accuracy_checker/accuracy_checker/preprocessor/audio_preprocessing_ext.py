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

import numpy as np

from ..config import BoolField, NumberField
from ..preprocessor import Preprocessor

class SpliceFrame(Preprocessor):
    __provider__ = 'audio_splice'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'frames': NumberField(optional=True, default=1, description="Number of frames to splice", value_type=int),
            'axis': NumberField(optional=True, default=2, description="Axis to splice frames along", value_type=int),
        })
        return parameters

    def configure(self):

        self.frames = self.get_value_from_config('frames')
        self.axis = self.get_value_from_config('axis')


    def process(self, image, annotation_meta=None):
        if self.frames > 1:
            seq = [image.data]
            for n in range(1, self.frames):
                tmp = np.zeros_like(image.data)
                tmp[:, :-n] = image.data[:, n:]
                seq.append(tmp)
            image.data = np.concatenate(seq, axis=self.axis)

        return image


class DitherFrame(Preprocessor):
    __provider__ = 'audio_dither'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'use_deterministic_dithering': BoolField(optional=True, default=True,
                                                     description="Deterministic dithering flag"),
            'dither': NumberField(optional=True, default=1e-5, description="Dithering factor", value_type=float),
        })
        return parameters

    def configure(self):
        self.use_deterministic_dithering = self.get_value_from_config('use_deterministic_dithering')
        self.dither = self.get_value_from_config('dither')

    def process(self, image, annotation_meta=None):
        if self.dither > 0 and not self.use_deterministic_dithering:
            image.data += self.dither * np.random.rand(*image.data.shape)
        return image


class PreemphFrame(Preprocessor):
    __provider__ = 'audio_preemph'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'preemph': NumberField(optional=True, default=0.97, description="Preemph factor", value_type=float),
        })
        return parameters

    def configure(self):
        self.preemph = self.get_value_from_config('preemph')

    def process(self, image, annotation_meta=None):
        if self.preemph != 0:
            image.data = np.concatenate((np.expand_dims(image.data[:, 0], axis=0),
                                         image.data[:, 1:] - self.preemph *  image.data[:, :-1]), axis=1)
        return image


class DitherSpectrum(Preprocessor):
    __provider__ = 'audio_spec_dither'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'use_deterministic_dithering': BoolField(optional=True, default=True,
                                                     description="Deterministic dithering flag"),
            'dither': NumberField(optional=True, default=1e-5, description="Dithering factor", value_type=float),
        })
        return parameters

    def configure(self):
        self.use_deterministic_dithering = self.get_value_from_config('use_deterministic_dithering')
        self.dither = self.get_value_from_config('dither')

    def process(self, image, annotation_meta=None):
        if self.dither > 0 and not self.use_deterministic_dithering:
            image.data = image.data + self.dither ** 2
        return image
