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
from .postprocessor import Postprocessor
from ..representation import FeaturesRegressionAnnotation, RegressionAnnotation, RegressionPrediction
from ..config import PathField, NumberField, BoolField
from ..data_readers import KaldiARKReader


class DeepFilterPostprocessor(Postprocessor):
    __provider__ = 'deep_filter'
    annotation_types = (RegressionAnnotation, FeaturesRegressionAnnotation)
    prediction_types = (RegressionPrediction, )

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'input_spec_file': PathField(description='path to ark file with input specification'),
            'time_pad': NumberField(
                optional=True, value_type=int, default=1, description='padding for time dim', min_value=1
            ),
            'feat_pad': NumberField(
                optional=True, value_type=int, default=1, description='padding for feat dim', min_value=1),
            'look_forward': BoolField(optional=True, default=False, description='allow filter looking forward')
        })
        return params

    def configure(self):
        self.time_pad = self.get_value_from_config('time_pad')
        self.feat_pad = self.get_value_from_config('feat_pad')
        self.is_look_forward = self.get_value_from_config('look_forward')
        input_spec_file = self.get_value_from_config('input_spec_file')
        mix_spec = KaldiARKReader.read_frames(input_spec_file)
        mix_spec  = next(iter(mix_spec.values())).astype(float).reshape(1, -1, 161, 2)
        self.stack_res = self.deep_filter(mix_spec)[..., 1:]

    def process_image(self, annotation, prediction):
        for ann, pred in zip(annotation, prediction):
            ann.value = self.apply_filter(ann.value)
            pred.value = self.apply_filter(pred.value)
        return annotation, prediction

    def deep_filter(self, data):
        bt, t, f, ch = data.shape
        if self.is_look_forward:
            pad_in = np.concatenate(
                [np.zeros([bt, self.time_pad, f, ch]), data, np.zeros([bt, self.time_pad, f, ch])], axis=1)
        else:
            pad_in = np.concatenate([np.zeros([bt, self.time_pad * 2, f, ch]), data], axis=1)
        pad_in = np.concatenate([np.zeros([bt, t + self.time_pad * 2, self.feat_pad, ch]), pad_in,
                                np.zeros([bt, t + self.time_pad * 2, self.feat_pad, ch])], axis=2)

        # slice time dim
        t_list = []
        for i in range(2 * self.time_pad + 1):
            t_list.append(pad_in[:, i:i + t])
        stack_res = np.stack(t_list, axis=1)

        # slice feat dim
        f_list = []
        for i in range(2 * self.feat_pad + 1):
            f_list.append(stack_res[:, :, :, i:i + f])
        stack_res = np.stack(f_list, axis=5).transpose((0, 1, 5, 4, 2, 3))
        return stack_res

    def apply_filter(self, data):
        est_d1 = data.astype(float).reshape(1, -1, 160, 18).transpose((0, 3, 1, 2))
        est_d1_pad = np.zeros((est_d1.shape[0], est_d1.shape[1], 1, est_d1.shape[3]), dtype=np.float32)
        d1 = np.concatenate((est_d1, est_d1_pad), axis=2)

        deep_filter = d1.reshape((d1.shape[0], 3, 3, 2, d1.shape[2], d1.shape[3]))

        a = self.stack_res[:, :, :, 0]
        b = self.stack_res[:, :, :, 1]
        c = deep_filter[:, :, :, 0]
        d = deep_filter[:, :, :, 1]
        est_real = (a * c - b * d).sum(1).sum(1)
        est_img = (a * d + b * c).sum(1).sum(1)
        est_spec = np.stack([est_real, est_img], axis=-1)

        est_spec_pad = np.zeros((est_spec.shape[0], est_spec.shape[1], 1, 2), dtype=np.float32)

        est_spec = np.concatenate((est_spec_pad, est_spec), axis=2)

        return est_spec

