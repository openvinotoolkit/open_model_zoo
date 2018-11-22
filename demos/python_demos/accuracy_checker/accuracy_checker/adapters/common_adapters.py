"""
 Copyright (c) 2018 Intel Corporation

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

from ..config import ConfigValidator, StringField
from ..representation import (ClassificationPrediction, DetectionPrediction, ReIdentificationPrediction,
                              SegmentationPrediction, CharacterRecognitionPrediction, ContainerPrediction,
                              RegressionPrediction, PointRegressionPrediction, MultilabelRecognitionPrediction)
from .adapter import Adapter


class ClassificationAdapter(Adapter):
    """
    Class for converting output of classification model to ClassificationPrediction representation
    """
    __provider__ = 'classification'

    def process(self, raw, identifiers=None):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
        Returns:
            list of ClassificationPrediction objects
        """
        prediction = raw[self.output_blob]
        prediction = np.reshape(prediction, (prediction.shape[0], -1))

        result = []
        for identifier, output in zip(identifiers, prediction):
            result.append(ClassificationPrediction(identifier, output))

        return result


class SegmentationAdapter(Adapter):
    __provider__ = 'segmentation'

    def process(self, raw, identifiers=None):
        prediction = raw[self.output_blob]

        result = []
        for identifier, output in zip(identifiers, prediction):
            result.append(SegmentationPrediction(identifier, output))

        return result


class TinyYOLOv1Adapter(Adapter):
    """
    Class for converting output of Tiny YOLO v1 model to DetectionPrediction representation
    """
    __provider__ = 'tiny_yolo_v1'

    def process(self, raw, identifiers=None):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
        Returns:
             list of DetectionPrediction objects
        """
        prediction = raw[self.output_blob]

        PROBABILITY_SIZE = 980
        CONFIDENCE_SIZE = 98
        BOXES_SIZE = 392

        CELLS_X, CELLS_Y = 7, 7
        CLASSES = 20
        OBJECTS_PER_CELL = 2

        result = []
        for identifier, output in zip(identifiers, prediction):
            assert PROBABILITY_SIZE + CONFIDENCE_SIZE + BOXES_SIZE == output.shape[0]

            probability, scale, boxes = np.split(output, [PROBABILITY_SIZE, PROBABILITY_SIZE + CONFIDENCE_SIZE])

            probability = np.reshape(probability, (CELLS_Y, CELLS_X, CLASSES))
            scale = np.reshape(scale, (CELLS_Y, CELLS_X, OBJECTS_PER_CELL))
            boxes = np.reshape(boxes, (CELLS_Y, CELLS_X, OBJECTS_PER_CELL, 4))

            confidence = np.zeros((CELLS_Y, CELLS_X, OBJECTS_PER_CELL, CLASSES + 4))
            for cls in range(CLASSES):
                confidence[:, :, 0, cls] = np.multiply(probability[:, :, cls], scale[:, :, 0])
                confidence[:, :, 1, cls] = np.multiply(probability[:, :, cls], scale[:, :, 1])

            labels, scores, x_mins, y_mins, x_maxs, y_maxs = [], [], [], [], [], []
            for i, j, k in np.ndindex((CELLS_X, CELLS_Y, OBJECTS_PER_CELL)):
                box = boxes[j, i, k]
                box = [(box[0] + i) / float(CELLS_X), (box[1] + j) / float(CELLS_Y), box[2] ** 2, box[3] ** 2]

                label = np.argmax(confidence[j, i, k, :CLASSES])
                score = confidence[j, i, k, label]

                labels.append(label + 1)
                scores.append(score)
                x_mins.append(box[0] - box[2] / 2.0)
                y_mins.append(box[1] - box[3] / 2.0)
                x_maxs.append(box[0] + box[2] / 2.0)
                y_maxs.append(box[1] + box[3] / 2.0)

            result.append(DetectionPrediction(identifier, labels, scores, x_mins, y_mins, x_maxs, y_maxs))
        return result


class ReidAdapter(Adapter):
    """
    Class for converting output of Reid model to ReIdentificationPrediction representation
    """
    __provider__ = 'reid'

    def configure(self):
        """
        Specifies parameters of config entry
        """
        self.grn_workaround = self.launcher_config.get("grn_workaround", True)

    def process(self, raw, identifiers=None):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
        Returns:
            list of ReIdentificationPrediction objects
        """
        prediction = raw[self.output_blob]

        if self.grn_workaround:
            # workaround: GRN layer
            prediction = self._grn_layer(prediction)

        return [ReIdentificationPrediction(identifier, embedding)
                for identifier, embedding in zip(identifiers, prediction)]

    @staticmethod
    def _grn_layer(prediction):
        GRN_BIAS = 0.000001
        sum_ = np.sum(prediction ** 2, axis=1)
        prediction = prediction / np.sqrt(sum_[:, np.newaxis] + GRN_BIAS)
        return prediction


class YoloV2Adapter(Adapter):
    """
    Class for converting output of YOLO v2 model to DetectionPrediction representation
    """
    __provider__ = 'yolo_v2'

    def process(self, raw, identifiers=None):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
        Returns:
            list of DetectionPrediction objects
        """
        predictions = raw[self.output_blob]

        def entry_index(w, h, n_coords, n_classes, pos, entry):
            row = pos // (w * h)
            col = pos % (w * h)
            return row * w * h * (n_classes + n_coords + 1) + entry * w * h + col

        cells_x, cells_y = 13, 13

        CLASSES = 20
        COORDS = 4
        NUM = 5
        ANCHORS = [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]

        result = []
        for identifier, prediction in zip(identifiers, predictions):
            labels, scores, x_mins, y_mins, x_maxs, y_maxs = [], [], [], [], [], []
            for y, x, n in np.ndindex((cells_y, cells_x, NUM)):
                index = n * cells_y * cells_x + y * cells_x + x

                box_index = entry_index(cells_x, cells_y, COORDS, CLASSES, index, 0)
                obj_index = entry_index(cells_x, cells_y, COORDS, CLASSES, index, COORDS)

                scale = prediction[obj_index]

                box = [
                    (x + prediction[box_index + 0 * (cells_y * cells_x)]) / cells_x,
                    (y + prediction[box_index + 1 * (cells_y * cells_x)]) / cells_y,
                    np.exp(prediction[box_index + 2 * (cells_y * cells_x)]) * ANCHORS[2 * n + 0] / cells_x,
                    np.exp(prediction[box_index + 3 * (cells_y * cells_x)]) * ANCHORS[2 * n + 1] / cells_y
                ]

                classes_prob = np.empty(CLASSES)
                for cls in range(CLASSES):
                    cls_index = entry_index(cells_x, cells_y, COORDS, CLASSES, index, COORDS + 1 + cls)
                    classes_prob[cls] = prediction[cls_index]

                classes_prob = classes_prob * scale

                label = np.argmax(classes_prob)

                labels.append(label + 1)
                scores.append(classes_prob[label])
                x_mins.append(box[0] - box[2] / 2.0)
                y_mins.append(box[1] - box[3] / 2.0)
                x_maxs.append(box[0] + box[2] / 2.0)
                y_maxs.append(box[1] + box[3] / 2.0)

            result.append(DetectionPrediction(identifier, labels, scores, x_mins, y_mins, x_maxs, y_maxs))

        return result


class LPRAdapter(Adapter):
    __provider__ = 'lpr'

    def process(self, raw, identifiers=None):
        predictions = raw[self.output_blob]
        result = []
        for identifier, output in zip(identifiers, predictions):
            decoded_out = self.decode(output.reshape(-1))
            result.append(CharacterRecognitionPrediction(identifier, decoded_out))
        return result

    def decode(self, outputs):
        decode_out = str()
        for output in outputs:
            if output == -1:
                break
            decode_out += str(self.label_map[output])
        return decode_out


class SSDAdapter(Adapter):
    """
    Class for converting output of SSD model to DetectionPrediction representation
    """
    __provider__ = 'ssd'

    def process(self, raw, identifiers=None):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
        Returns:
            list of DetectionPrediction objects
        """
        prediction_batch = raw[self.output_blob]
        prediction_count = prediction_batch.shape[2]
        prediction_batch = prediction_batch.reshape(prediction_count, -1)
        prediction_batch = self.remove_empty_detections(prediction_batch)


        result = []
        for batch_index, identifier in enumerate(identifiers):
            prediction_mask = np.where(prediction_batch[:, 0] == batch_index)
            detections = prediction_batch[prediction_mask]
            detections = detections[:, 1::]
            result.append(DetectionPrediction(identifier, *zip(*detections)))

        return result

    @staticmethod
    def remove_empty_detections(prediction_blob):
        ind = prediction_blob[:, 0]
        ind_ = np.where(ind == -1)[0]
        m = ind_[0] if ind_.size else prediction_blob.shape[0]
        return prediction_blob[:m, :]


class FacePersonDetectionAdapterConfig(ConfigValidator):
    type = StringField()
    face_out = StringField()
    person_out = StringField()


class FacePersonAdapter(Adapter):
    __provider__ = 'face_person_detection'

    def validate_config(self):
        face_person_detection_adapter_config = FacePersonDetectionAdapterConfig('FacePersonDetection_Config')
        face_person_detection_adapter_config.validate(self.launcher_config)

    def configure(self):
        self.face_detection_out = self.launcher_config['face_out']
        self.person_detection_out = self.launcher_config['person_out']
        self.face_adapter = SSDAdapter(self.launcher_config, self.label_map, self.face_detection_out)
        self.person_adapter = SSDAdapter(self.launcher_config, self.label_map, self.person_detection_out)

    def process(self, raw, identifiers=None):
        face_batch_result = self.face_adapter(raw, identifiers)
        person_batch_result = self.person_adapter(raw, identifiers)
        result = [ContainerPrediction({self.face_detection_out: face_result, self.person_detection_out: person_result})
                  for face_result, person_result in zip(face_batch_result, person_batch_result)]
        return result


class HeadPoseEstimatorAdapterConfig(ConfigValidator):
    type = StringField()
    angle_yaw = StringField()
    angle_pitch = StringField()
    angle_roll = StringField()


class HeadPoseEstimatorAdapter(Adapter):
    """
    Class for converting output of HeadPoseEstimator to HeadPosePrediction representation
    """
    __provider__ = 'head_pose'

    def validate_config(self):
        head_pose_estimator_adapter_config = HeadPoseEstimatorAdapterConfig('HeadPoseEstimator_Config')
        head_pose_estimator_adapter_config.validate(self.launcher_config)

    def configure(self):
        """
        Specifies parameters of config entry
        """
        self.angle_yaw = self.launcher_config['angle_yaw']
        self.angle_pitch = self.launcher_config['angle_pitch']
        self.angle_roll = self.launcher_config['angle_roll']

    def process(self, raw, identifiers=None):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
        Returns:
                list of ContainerPrediction objects
        """
        result = []
        for identifier, yaw, pitch, roll in zip(
                identifiers, raw[self.angle_yaw], raw[self.angle_pitch], raw[self.angle_roll]):
            prediction = ContainerPrediction({'angle_yaw': RegressionPrediction(identifier, yaw[0]),
                                              'angle_pitch': RegressionPrediction(identifier, pitch[0]),
                                              'angle_roll': RegressionPrediction(identifier, roll[0])})
            result.append(prediction)

        return result


class VehicleAttributesRecognitionAdapterConfig(ConfigValidator):
    type = StringField()
    color_out = StringField()
    type_out = StringField()


class VehicleAttributesRecognitionAdapter(Adapter):
    __provider__ = 'vehicle_attributes'

    def validate_config(self):
        attributes_recognition_adapter_config = VehicleAttributesRecognitionAdapterConfig(
            'VehicleAttributesRecognition_Config')
        attributes_recognition_adapter_config.validate(self.launcher_config)

    def configure(self):
        """
        Specifies parameters of config entry
        """
        self.color_out = self.launcher_config['color_out']
        self.type_out = self.launcher_config['type_out']

    def process(self, raw, identifiers=None):
        res = []
        for identifier, colors, types in zip(identifiers,
                                             raw[self.color_out], raw[self.type_out]):
            res.append(ContainerPrediction({'color': ClassificationPrediction(identifier, colors.reshape(-1)),
                                            'type': ClassificationPrediction(identifier, types.reshape(-1))}))
        return res


class AgeGenderAdapterConfig(ConfigValidator):
    type = StringField()
    age_out = StringField()
    gender_out = StringField()


class AgeGenderAdapter(Adapter):
    __provider__ = 'age_gender'

    def configure(self):
        self.age_out = self.launcher_config['age_out']
        self.gender_out = self.launcher_config['gender_out']

    def validate_config(self):
        age_gender_adapter_config = AgeGenderAdapterConfig('AgeGender_Config')
        age_gender_adapter_config.validate(self.launcher_config)

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

    def process(self, raw, identifiers=None):
        result = []
        for identifier, age, gender in zip(identifiers, raw[self.age_out], raw[self.gender_out]):
            gender = gender.reshape(-1)
            age = age.reshape(-1)[0]*100
            gender_rep = ClassificationPrediction(identifier, gender)
            age_claas_rep = ClassificationPrediction(identifier, self.get_age_scores(age))
            age_error_rep = RegressionPrediction(identifier, age)
            result.append(ContainerPrediction({'gender': gender_rep, 'age_classification': age_claas_rep,
                                               'age_error': age_error_rep}))
        return result


class LandmarksRegressionAdapter(Adapter):
    __provider__ = 'landmarks_regression'

    def process(self, raw, identifiers=None):
        res = []
        for identifier, values in zip(identifiers, raw[self.output_blob]):
            x_values, y_values = values[::2], values[1::2]
            res.append(PointRegressionPrediction(identifier, x_values.reshape(-1), y_values.reshape(-1)))
        return res


class PersonAttributesAdapter(Adapter):
    __provider__ = 'person_attributes'

    def process(self, raw, identifiers=None):
        result = []
        for identifier, multi_label in zip(identifiers, raw[self.output_blob]):
            multi_label[np.where(multi_label >= 0.5)] = 1.
            multi_label[np.where(multi_label < 0.5)] = 0.

            result.append(MultilabelRecognitionPrediction(identifier, multi_label))
        return result
