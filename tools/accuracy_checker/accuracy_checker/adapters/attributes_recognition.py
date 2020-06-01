"""
Copyright (c) 2019 Intel Corporation

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

from ..adapters import Adapter
from ..config import ConfigValidator, StringField, PathField
from ..representation import (
    ContainerPrediction,
    RegressionPrediction,
    ClassificationPrediction,
    FacialLandmarksPrediction,
    MultiLabelRecognitionPrediction,
    GazeVectorPrediction,
    FacialLandmarks3DPrediction
)


class HeadPoseEstimatorAdapter(Adapter):
    """
    Class for converting output of HeadPoseEstimator to HeadPosePrediction representation
    """
    __provider__ = 'head_pose'
    prediction_types = (RegressionPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'angle_yaw': StringField(description="Output layer name for yaw angle."),
            'angle_pitch': StringField(description="Output layer name for pitch angle."),
            'angle_roll': StringField(description="Output layer name for roll angle.")
        })
        return parameters

    def validate_config(self):
        super().validate_config(on_extra_argument=ConfigValidator.ERROR_ON_EXTRA_ARGUMENT)

    def configure(self):
        """
        Specifies parameters of config entry
        """
        self.angle_yaw = self.get_value_from_config('angle_yaw')
        self.angle_pitch = self.get_value_from_config('angle_pitch')
        self.angle_roll = self.get_value_from_config('angle_roll')

    def process(self, raw, identifiers=None, frame_meta=None):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
            frame_meta: list of meta information about each frame
        Returns:
                list of ContainerPrediction objects
        """
        result = []
        raw_output = self._extract_predictions(raw, frame_meta)
        for identifier, yaw, pitch, roll in zip(
                identifiers,
                raw_output[self.angle_yaw],
                raw_output[self.angle_pitch],
                raw_output[self.angle_roll]
        ):
            prediction = ContainerPrediction({
                'angle_yaw': RegressionPrediction(identifier, yaw[0]),
                'angle_pitch': RegressionPrediction(identifier, pitch[0]),
                'angle_roll': RegressionPrediction(identifier, roll[0])
            })
            result.append(prediction)

        return result


class VehicleAttributesRecognitionAdapter(Adapter):
    __provider__ = 'vehicle_attributes'
    prediction_types = (ClassificationPrediction,)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'color_out': StringField(description="Vehicle color attribute output layer name."),
            'type_out': StringField(description="Vehicle type attribute output layer name.")
        })
        return parameters

    def validate_config(self):
        super().validate_config(on_extra_argument=ConfigValidator.ERROR_ON_EXTRA_ARGUMENT)

    def configure(self):
        """
        Specifies parameters of config entry
        """
        self.color_out = self.get_value_from_config('color_out')
        self.type_out = self.get_value_from_config('type_out')

    def process(self, raw, identifiers=None, frame_meta=None):
        res = []
        raw_output = self._extract_predictions(raw, frame_meta)
        for identifier, colors, types in zip(identifiers, raw_output[self.color_out], raw_output[self.type_out]):
            res.append(ContainerPrediction({
                'color': ClassificationPrediction(identifier, colors.reshape(-1)),
                'type': ClassificationPrediction(identifier, types.reshape(-1))
            }))

        return res


class AgeGenderAdapter(Adapter):
    __provider__ = 'age_gender'
    prediction_types = (ClassificationPrediction, RegressionPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'age_out': StringField(description="Output layer name for age recognition."),
            'gender_out': StringField(description="Output layer name for gender recognition.")
        })
        return parameters

    def configure(self):
        self.age_out = self.get_value_from_config('age_out')
        self.gender_out = self.get_value_from_config('gender_out')

    def validate_config(self):
        super().validate_config(on_extra_argument=ConfigValidator.ERROR_ON_EXTRA_ARGUMENT)

    @staticmethod
    def get_age_scores(age):
        age_scores = np.zeros(4)
        if age < 19:
            age_scores[0] = 1
            return age_scores
        if age < 36:
            age_scores[1] = 1
            return age_scores
        if age < 66:
            age_scores[2] = 1
            return age_scores
        age_scores[3] = 1
        return age_scores

    def process(self, raw, identifiers=None, frame_meta=None):
        result = []
        raw_output = self._extract_predictions(raw, frame_meta)
        for identifier, age, gender in zip(identifiers, raw_output[self.age_out], raw_output[self.gender_out]):
            gender = gender.reshape(-1)
            age = age.reshape(-1)[0]*100
            gender_rep = ClassificationPrediction(identifier, gender)
            age_class_rep = ClassificationPrediction(identifier, self.get_age_scores(age))
            age_error_rep = RegressionPrediction(identifier, age)
            result.append(ContainerPrediction({
                'gender': gender_rep, 'age_classification': age_class_rep, 'age_error': age_error_rep
            }))

        return result


class LandmarksRegressionAdapter(Adapter):
    __provider__ = 'landmarks_regression'
    prediction_types = (FacialLandmarksPrediction, )

    def process(self, raw, identifiers=None, frame_meta=None):
        res = []
        raw_output = self._extract_predictions(raw, frame_meta)
        for identifier, values in zip(identifiers, raw_output[self.output_blob]):
            x_values, y_values = values[::2], values[1::2]
            res.append(FacialLandmarksPrediction(identifier, x_values.reshape(-1), y_values.reshape(-1)))

        return res


class PersonAttributesAdapter(Adapter):
    __provider__ = 'person_attributes'
    prediction_types = (MultiLabelRecognitionPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'attributes_recognition_out': StringField(
                description="Output layer name for attributes recognition.", optional=True
            )
        })
        return parameters

    def validate_config(self):
        super().validate_config(on_extra_argument=ConfigValidator.IGNORE_ON_EXTRA_ARGUMENT)

    def configure(self):
        self.attributes_recognition_out = self.launcher_config.get('attributes_recognition_out', self.output_blob)

    def process(self, raw, identifiers=None, frame_meta=None):
        result = []
        raw_output = self._extract_predictions(raw, frame_meta)
        self.attributes_recognition_out = self.attributes_recognition_out or self.output_blob
        for identifier, multi_label in zip(identifiers, raw_output[self.attributes_recognition_out]):
            multi_label[multi_label > 0.5] = 1.
            multi_label[multi_label <= 0.5] = 0.

            result.append(MultiLabelRecognitionPrediction(identifier, multi_label.reshape(-1)))

        return result


class GazeEstimationAdapter(Adapter):
    __provider__ = 'gaze_estimation'
    prediction_types = (GazeVectorPrediction, )

    def process(self, raw, identifiers=None, frame_meta=None):
        result = []
        raw_output = self._extract_predictions(raw, frame_meta)
        for identifier, output in zip(identifiers, raw_output[self.output_blob]):
            result.append(GazeVectorPrediction(identifier, output))

        return result


class PRNetAdapter(Adapter):
    __provider__ = 'prnet'
    landmarks_uv = np.array([
        [15, 22, 26, 32, 45, 67, 91, 112, 128, 143, 164, 188, 210, 223, 229, 233, 240, 58, 71, 85, 97, 106, 149, 158,
         170, 184, 197, 128, 128, 128, 128, 117, 122, 128, 133, 138, 78, 86, 95, 102, 96, 87, 153, 160, 169, 177, 168,
         159, 108, 116, 124, 128, 131, 139, 146, 137, 132, 128, 123, 118, 110, 122, 128, 133, 145, 132, 128, 123],
        [96, 118, 141, 165, 183, 190, 188, 187, 193, 187, 188, 190, 183, 165, 141, 118, 96, 49, 42, 39, 40, 42, 42, 40,
         39, 42, 49, 59, 73, 86, 96, 111, 113, 115, 113, 111, 67, 60, 61, 65, 68, 69, 65, 61, 60, 67, 69, 68, 142, 131,
         127, 128, 127, 131, 142, 148, 150, 150, 150, 148, 141, 135, 134, 135, 142, 143, 142, 143]])

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'landmarks_ids_file': PathField(
                description='text file with landmarks indexes in 3D face dense pose mask',
                optional=True
            )
        })
        return params

    def configure(self):
        self.landmarks_ids_file = self.get_value_from_config('landmarks_ids_file')
        if self.landmarks_ids_file:
            self.landmarks_uv = np.loadtxt(str(self.landmarks_ids_file)).astype(int)

    def process(self, raw, identifiers=None, frame_meta=None):
        result = []
        raw_output = self._extract_predictions(raw, frame_meta)
        for identifier, pos, meta in zip(identifiers, raw_output[self.output_blob], frame_meta):
            input_shape = next(iter(meta['input_shape'].values()))
            if input_shape[1] == 3:
                height, width = input_shape[2], input_shape[3]
                pos = np.transpose(pos, (1, 2, 0))
            else:
                height, width = input_shape[1], input_shape[2]
            pos *= (height * 1.1)
            vertices = np.reshape(pos, [-1, 3]).T
            if 'transform_matrix' in meta:
                z = vertices[2, :].copy() / meta['transform_matrix'][0, 0]
                vertices[2, :] = 1
                vertices = np.dot(np.linalg.inv(meta['transform_matrix']), vertices)
                vertices = np.vstack((vertices[:2, :], z))
            pos = np.reshape(vertices.T, [height, width, 3])
            kpt = pos[self.landmarks_uv[1], self.landmarks_uv[0], :]
            x_values, y_values, z_values = kpt.T
            z_values -= z_values.mean()
            result.append(FacialLandmarks3DPrediction(identifier, x_values, y_values, z_values))

        return result
