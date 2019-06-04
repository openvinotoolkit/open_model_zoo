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
from ..config import ConfigValidator, StringField, NumberField
from ..representation import DetectionPrediction, ActionDetectionPrediction, ContainerPrediction


class ActionDetection(Adapter):
    __provider__ = 'action_detection'
    prediction_types = (ActionDetectionPrediction, DetectionPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'priorbox_out': StringField(description="Name of layer containing prior boxes in SSD format."),
            'loc_out': StringField(description="Name of layer containing box coordinates in SSD format."),
            'main_conf_out': StringField(description="Name of layer containing detection confidences."),
            'add_conf_out_prefix': StringField(
                description="Prefix for generation name of layers containing action confidences "
                            "if topology has several following layers or layer name."
            ),
            'add_conf_out_count': NumberField(
                optional=True, min_value=1,
                description="Number of layers with action confidences (optional, you can not provide this argument "
                            "if action confidences contained in one layer)."
            ),
            'num_action_classes': NumberField(description="Number classes for action recognition."),
            'detection_threshold': NumberField(
                optional=True, value_type=float, min_value=0, max_value=1, default=0,
                description="Minimal detection confidences level for valid detections."),
            'action_scale': NumberField(
                optional=True, value_type=float, default=3, description="Scale for correct action score calculation."
            ),
            'action_confidence_threshold': NumberField(
                optional=True, value_type=float, min_value=0, max_value=1, default=0,
                description="Action confidence threshold."
            )
        })

        return parameters

    def validate_config(self):
        super().validate_config(on_extra_argument=ConfigValidator.WARN_ON_EXTRA_ARGUMENT)

    def configure(self):
        self.priorbox_out = self.get_value_from_config('priorbox_out')
        self.loc_out = self.get_value_from_config('loc_out')
        self.main_conf_out = self.get_value_from_config('main_conf_out')
        self.num_action_classes = self.get_value_from_config('num_action_classes')
        self.detection_threshold = self.get_value_from_config('detection_threshold')
        add_conf_out_count = self.get_value_from_config('add_conf_out_count')
        add_conf_out_prefix = self.get_value_from_config('add_conf_out_prefix')
        self.action_threshold = self.get_value_from_config('action_confidence_threshold')
        self.action_scale = self.get_value_from_config('action_scale')
        if add_conf_out_count is None:
            self.add_conf_outs = [add_conf_out_prefix]
        else:
            self.add_conf_outs = []
            for num in np.arange(start=1, stop=add_conf_out_count + 1):
                self.add_conf_outs.append('{}{}'.format(add_conf_out_prefix, num))

    def process(self, raw, identifiers=None, frame_meta=None):
        result = []
        raw_outputs = self._extract_predictions(raw, frame_meta)
        prior_boxes = raw_outputs[self.priorbox_out][0][0].reshape(-1, 4)
        prior_variances = raw_outputs[self.priorbox_out][0][1].reshape(-1, 4)
        for batch_id, identifier in enumerate(identifiers):
            labels, class_scores, x_mins, y_mins, x_maxs, y_maxs, main_scores = self.prepare_detection_for_id(
                batch_id, raw_outputs, prior_boxes, prior_variances
            )
            action_prediction = ActionDetectionPrediction(
                identifier, labels, class_scores, main_scores, x_mins, y_mins, x_maxs, y_maxs
            )
            person_prediction = DetectionPrediction(
                identifier, [1] * len(labels), main_scores, x_mins, y_mins, x_maxs, y_maxs
            )
            result.append(ContainerPrediction({
                'action_prediction': action_prediction, 'class_agnostic_prediction': person_prediction
            }))

        return result

    def prepare_detection_for_id(self, batch_id, raw_outputs, prior_boxes, prior_variances, default_label=0):
        num_detections = raw_outputs[self.loc_out][batch_id].size // 4
        locs = raw_outputs[self.loc_out][batch_id].reshape(-1, 4)
        main_conf = raw_outputs[self.main_conf_out][batch_id].reshape(num_detections, -1)
        add_confs = [raw_outputs[layer][batch_id].reshape(-1, self.num_action_classes) for layer in self.add_conf_outs]
        anchors_num = len(add_confs)
        labels, class_scores, x_mins, y_mins, x_maxs, y_maxs, main_scores = [], [], [], [], [], [], []
        for index in range(num_detections):
            if main_conf[index, 1] < self.detection_threshold:
                continue

            x_min, y_min, x_max, y_max = self.decode_box(prior_boxes[index], prior_variances[index], locs[index])
            action_confs = add_confs[index % anchors_num][index // anchors_num]
            exp_action_confs = np.exp(self.action_scale * action_confs)
            sum_exp_conf = np.sum(exp_action_confs)
            action_label = np.argmax(action_confs)
            action_score = exp_action_confs[action_label] / sum_exp_conf

            if action_score < self.action_threshold:
                action_label = default_label
                action_score = 0
            labels.append(action_label)
            class_scores.append(action_score)
            x_mins.append(x_min)
            y_mins.append(y_min)
            x_maxs.append(x_max)
            y_maxs.append(y_max)
            main_scores.append(main_conf[index, 1])

        return labels, class_scores, x_mins, y_mins, x_maxs, y_maxs, main_scores

    @staticmethod
    def decode_box(prior, var, deltas):
        prior_width = prior[2] - prior[0]
        prior_height = prior[3] - prior[1]
        prior_center_x = (prior[0] + prior[2]) / 2.
        prior_center_y = (prior[1] + prior[3]) / 2.

        decoded_box_center_x = var[0] * deltas[0] * prior_width + prior_center_x
        decoded_box_center_y = var[1] * deltas[1] * prior_height + prior_center_y
        decoded_box_width = np.exp(var[2] * deltas[2]) * prior_width
        decoded_box_height = np.exp(var[3] * deltas[3]) * prior_height

        decoded_xmin = decoded_box_center_x - decoded_box_width / 2.
        decoded_ymin = decoded_box_center_y - decoded_box_height / 2.
        decoded_xmax = decoded_box_center_x + decoded_box_width / 2.
        decoded_ymax = decoded_box_center_y + decoded_box_height / 2.

        return decoded_xmin, decoded_ymin, decoded_xmax, decoded_ymax
