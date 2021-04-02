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
from .base_representation import BaseRepresentation
from ..data_readers import BaseReader


class RegressionRepresentation(BaseRepresentation):
    def __init__(self, identifier='', value=None):
        super().__init__(identifier)
        self.value = value


class RegressionAnnotation(RegressionRepresentation):
    pass


class RegressionPrediction(RegressionRepresentation):
    def to_annotation(self, **kwargs):
        return RegressionAnnotation(self.identifier, self.value)


class GazeVectorRepresentation(RegressionRepresentation):
    def __init__(self, identifier='', value=None):
        if value is None:
            value = np.array([])
        super().__init__(identifier, value)


class GazeVectorAnnotation(GazeVectorRepresentation):
    pass


class GazeVectorPrediction(GazeVectorRepresentation):
    def to_annotation(self, **kwargs):
        return GazeVectorAnnotation(self.identifier, self.value)


class FacialLandmarksRepresentation(BaseRepresentation):
    def __init__(self, identifier='', x_values=None, y_values=None):
        super().__init__(identifier)
        self.x_values = x_values if x_values is not None else []
        self.y_values = y_values if y_values is not None else []


class FacialLandmarksAnnotation(FacialLandmarksRepresentation):
    @property
    def interocular_distance(self):
        left_eye = [
            np.mean(self.x_values[self.metadata['left_eye']]),
            np.mean(self.y_values[self.metadata['left_eye']])
        ]
        right_eye = [
            np.mean(self.x_values[self.metadata['right_eye']]),
            np.mean(self.y_values[self.metadata['right_eye']])
        ]

        return np.linalg.norm((np.subtract(left_eye, right_eye)))


class FacialLandmarksPrediction(FacialLandmarksRepresentation):
    pass


class FacialLandmarks3DRepresentation(BaseRepresentation):
    def __init__(self, identifier='', x_values=None, y_values=None, z_values=None):
        super().__init__(identifier)
        self.x_values = x_values if x_values is not None else []
        self.y_values = y_values if y_values is not None else []
        self.z_values = z_values if z_values is not None else []


class FacialLandmarks3DAnnotation(FacialLandmarks3DRepresentation, FacialLandmarksAnnotation):
    def __init__(self, identifier='', x_values=None, y_values=None, z_values=None, face_mask=None):
        super().__init__(identifier, x_values, y_values, z_values)
        self.face_mask = face_mask

    def normalization_coef(self, is_2d=False):
        if self.face_mask is None:
            min_x, max_x = np.min(self.x_values), np.max(self.x_values)
            min_y, max_y = np.min(self.y_values), np.max(self.y_values)
            min_z, max_z = np.min(self.z_values), np.max(self.z_values)
        else:
            face_vertices_x = self.x_values[self.face_mask > 0]
            face_vertices_y = self.y_values[self.face_mask > 0]
            face_vertices_z = self.x_values[self.face_mask > 0]
            min_x, max_x = np.min(face_vertices_x), np.max(face_vertices_x)
            min_y, max_y = np.min(face_vertices_y), np.max(face_vertices_y)
            min_z, max_z = np.min(face_vertices_z), np.max(face_vertices_z)
        if is_2d:
            return np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
        return np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2 + (max_z - min_z) ** 2)


class FacialLandmarks3DPrediction(FacialLandmarks3DRepresentation):
    pass


class FeaturesRegressionAnnotation(BaseRepresentation):
    def __init__(self, identifier, value_file, dict_features=False, is_bin=False, bin_dtype='float32'):
        super().__init__(identifier)
        self.value_file = value_file
        if not dict_features:
            self._reader_config = {'type': 'numpy_txt_reader'} if not is_bin else {
                "type": 'numpy_bin_reader', "dtype": bin_dtype
            }
        else:
            self._reader_config = {'type': 'numpy_dict_reader'}
        self._value = None

    @property
    def value(self):
        if self._value is None:
            data_source = self.metadata.get('additional_data_source')
            if data_source is None:
                data_source = self.metadata['data_source']
            reader = BaseReader.provide(self._reader_config['type'], data_source, self._reader_config)
            return reader.read(self.value_file)
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

class NiftiRegressionAnnotation(BaseRepresentation):
    def __init__(self, identifier, value_file, mask_channels_first=False, to_4D=True,
                 multi_frame=False, separator='#', frame_axis=-1):
        super().__init__(identifier)
        self.value_file = value_file
        self._reader_config = {'type': 'nifti_reader', 'to_4D': to_4D, 'channels_first': mask_channels_first,
                               'multi_frame': multi_frame, 'separator': separator, 'frame_axis': frame_axis}
        self._value = None

    @property
    def value(self):
        if self._value is None:
            data_source = self.metadata.get('additional_data_source')
            if data_source is None:
                data_source = self.metadata['data_source']
            reader = BaseReader.provide(self._reader_config['type'], data_source, self._reader_config)
            return reader.read(self.value_file)
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
