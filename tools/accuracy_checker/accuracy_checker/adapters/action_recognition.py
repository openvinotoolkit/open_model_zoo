"""
Copyright (c) 2018-2020 Intel Corporation

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

import re
import numpy as np

from ..adapters import Adapter
from ..config import ConfigValidator, StringField, NumberField, BoolField, ListField
from ..representation import DetectionPrediction, ActionDetectionPrediction, ContainerPrediction
from ..utils import contains_all


class ActionDetection(Adapter):
    __provider__ = 'action_detection'
    prediction_types = (ActionDetectionPrediction, DetectionPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'loc_out': StringField(description="Name of layer containing box coordinates in SSD format."),
            'main_conf_out': StringField(description="Name of layer containing detection confidences."),
            'multihead_net': BoolField(
                optional=True, default=False,
                description="Whether to configure for the multi head network architecture."
            ),
            'priorbox_out': StringField(
                optional=True, description="Name of layer containing prior boxes in SSD format."
            ),
            'add_conf_out_prefix': StringField(
                description="Prefix for generation name of layers containing action confidences "
                            "if topology has several following layers or layer name."
            ),
            'add_conf_out_suffix': StringField(
                optional=True,
                description="Suffix for generation name of layers containing action confidences "
                            "if topology has several following layers or layer name."
            ),
            'head_sizes': ListField(
                optional=True, value_type=NumberField(value_type=int, min_value=1, description="Head size"),
                description="Number of anchors for each SSD head."
            ),
            'head_scales': ListField(
                optional=True, value_type=NumberField(value_type=int, min_value=0, description="Network head scales"),
                description="Variances to decode SSD prediction."
            ),
            'anchors': ListField(
                optional=True, value_type=ListField(value_type=ListField(value_type=NumberField(
                    value_type=float, min_value=0.0))),
                description="SSD anchors."
            ),
            'variance': ListField(
                optional=True, value_type=NumberField(value_type=float, min_value=0.0, description="SSD variance"),
                description="Variances to decode SSD prediction."
            ),
            'in_sizes': ListField(
                optional=True, value_type=NumberField(value_type=int, min_value=0, description="Network input sizes"),
                description="Variances to decode SSD prediction."
            ),
            'add_conf_out_count': NumberField(
                optional=True, min_value=1, value_type=int,
                description="Number of layers with action confidences (optional, you can not provide this argument "
                            "if action confidences contained in one layer)."
            ),
            'num_action_classes': NumberField(description="Number classes for action recognition.", value_type=int),
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
        self.multihead = self.get_value_from_config('multihead_net')
        self.loc_out = self.get_value_from_config('loc_out')
        self.main_conf_out = self.get_value_from_config('main_conf_out')
        self.num_action_classes = self.get_value_from_config('num_action_classes')
        self.detection_threshold = self.get_value_from_config('detection_threshold')
        self.action_threshold = self.get_value_from_config('action_confidence_threshold')
        self.action_scale = self.get_value_from_config('action_scale')
        add_conf_out_prefix = self.get_value_from_config('add_conf_out_prefix')
        self.outputs_verified = False

        if self.multihead:
            self.in_sizes = self.get_value_from_config('in_sizes')
            self.variance = self.get_value_from_config('variance')
            self.head_sizes = self.get_value_from_config('head_sizes')
            self.head_scales = self.get_value_from_config('head_scales')
            self.anchors = self.get_value_from_config('anchors')
            add_conf_out_suffix = self.get_value_from_config('add_conf_out_suffix')

            self.add_conf_outs = []
            self.glob_layer_id_map = []
            for head_id in range(len(self.head_sizes)):
                glob_layer_ids = []
                for anchor_id in np.arange(start=1, stop=self.head_sizes[head_id] + 1):
                    self.add_conf_outs.append(
                        '{}{}{}{}'.format(add_conf_out_prefix, head_id + 1, add_conf_out_suffix, anchor_id)
                    )
                    glob_layer_ids.append(len(self.add_conf_outs) - 1)

                self.glob_layer_id_map.append(glob_layer_ids)
        else:
            self.priorbox_out = self.get_value_from_config('priorbox_out')
            add_conf_out_count = self.get_value_from_config('add_conf_out_count')
            if add_conf_out_count is None:
                self.add_conf_outs = [add_conf_out_prefix]
                self.head_sizes = [1]
                self.glob_layer_id_map = [[0]]
            else:
                self.add_conf_outs = []
                for num in np.arange(start=1, stop=add_conf_out_count + 1):
                    self.add_conf_outs.append('{}{}'.format(add_conf_out_prefix, num))
                self.head_sizes = [add_conf_out_count]
                self.glob_layer_id_map = [list(range(add_conf_out_count))]

    def process(self, raw, identifiers, frame_meta):
        result = []
        raw_outputs = self._extract_predictions(raw, frame_meta)
        if not self.outputs_verified:
            self._get_output_names(raw_outputs)
        prior_boxes = raw_outputs[self.priorbox_out][0][0].reshape(-1, 4) if not self.multihead else None
        prior_variances = raw_outputs[self.priorbox_out][0][1].reshape(-1, 4) if not self.multihead else None

        head_shifts = self.estimate_head_shifts(raw_outputs, self. head_sizes, self.add_conf_outs, self.multihead)

        for batch_id, identifier in enumerate(identifiers):
            labels, class_scores, x_mins, y_mins, x_maxs, y_maxs, main_scores = self.prepare_detection_for_id(
                batch_id, raw_outputs, prior_boxes, prior_variances, head_shifts
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

    def prepare_detection_for_id(self, batch_id, raw_outputs, prior_boxes, prior_variances, head_shifts,
                                 default_label=0):
        num_detections = raw_outputs[self.loc_out][batch_id].size // 4
        locs = raw_outputs[self.loc_out][batch_id].reshape(-1, 4)
        main_conf = raw_outputs[self.main_conf_out][batch_id].reshape(num_detections, -1)

        add_confs = [raw_outputs[layer][batch_id] for layer in self.add_conf_outs]
        if self.multihead:
            spatial_sizes = [layer.shape[1:] for layer in add_confs]
            add_confs = [layer.reshape(self.num_action_classes, -1) for layer in add_confs]
        else:
            add_confs = [layer.reshape(-1, self.num_action_classes) for layer in add_confs]

        labels, class_scores, x_mins, y_mins, x_maxs, y_maxs, main_scores = [], [], [], [], [], [], []
        for index in range(num_detections):
            if main_conf[index, 1] < self.detection_threshold:
                continue

            head_id = self.get_head_id(index, head_shifts)
            head_anchors_num = self.head_sizes[head_id]
            head_index = index - head_shifts[head_id]
            head_anchor_id = head_index % head_anchors_num
            glob_anchor_id = self.glob_layer_id_map[head_id][head_anchor_id]
            head_spatial_pos = head_index // head_anchors_num

            prior_box_data = self.generate_prior_box(
                head_spatial_pos, self.head_scales[head_id], self.anchors[head_id][head_anchor_id],
                self.in_sizes, spatial_sizes[glob_anchor_id]) if self.multihead else prior_boxes[index]
            prior_variance_data = self.variance if self.multihead else prior_variances[index]
            bbox_loc_data = locs[index][[1, 0, 3, 2]] if self.multihead else locs[index]
            x_min, y_min, x_max, y_max = self.decode_box(prior_box_data, prior_variance_data, bbox_loc_data)

            add_confs_data = add_confs[glob_anchor_id]
            action_confs = add_confs_data[:, head_spatial_pos] if self.multihead else add_confs_data[head_spatial_pos]
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
        prior_center_x = 0.5 * (prior[0] + prior[2])
        prior_center_y = 0.5 * (prior[1] + prior[3])

        decoded_box_center_x = var[0] * deltas[0] * prior_width + prior_center_x
        decoded_box_center_y = var[1] * deltas[1] * prior_height + prior_center_y
        decoded_box_width = np.exp(var[2] * deltas[2]) * prior_width
        decoded_box_height = np.exp(var[3] * deltas[3]) * prior_height

        decoded_xmin = decoded_box_center_x - 0.5 * decoded_box_width
        decoded_ymin = decoded_box_center_y - 0.5 * decoded_box_height
        decoded_xmax = decoded_box_center_x + 0.5 * decoded_box_width
        decoded_ymax = decoded_box_center_y + 0.5 * decoded_box_height

        return decoded_xmin, decoded_ymin, decoded_xmax, decoded_ymax

    @staticmethod
    def estimate_head_shifts(raw_outputs, head_sizes, add_conf_outs, multihead_net):
        layer_id = 0
        head_shift = 0
        head_shifts = [0]
        for head_size in head_sizes:
            for _ in range(head_size):
                layer = add_conf_outs[layer_id]
                layer_shape = raw_outputs[layer][0].shape
                layer_size = np.prod(layer_shape[1:]) if multihead_net else np.prod(layer_shape[:2])
                head_shift += layer_size
                layer_id += 1

            head_shifts.append(head_shift)

        return head_shifts

    @staticmethod
    def get_head_id(index, head_shifts):
        head_id = 0
        while index >= head_shifts[head_id + 1]:
            head_id += 1
        return head_id

    @staticmethod
    def generate_prior_box(pos, step, anchor, image_size, blob_size):
        image_height, image_width = image_size
        anchor_height, anchor_width = anchor

        row = pos // blob_size[1]
        col = pos % blob_size[1]

        center_x = (col + 0.5) * step
        center_y = (row + 0.5) * step

        normalized_bbox = [
            (center_x - 0.5 * anchor_width) / float(image_width),
            (center_y - 0.5 * anchor_height) / float(image_height),
            (center_x + 0.5 * anchor_width) / float(image_width),
            (center_y + 0.5 * anchor_height) / float(image_height)
        ]

        return normalized_bbox

    def _get_output_names(self, raw_outputs):
        loc_out_regex = re.compile(self.loc_out)
        main_conf_out_regex = re.compile(self.main_conf_out)

        def find_layer(regex, output_name, all_outputs):
            suitable_layers = [layer_name for layer_name in all_outputs if regex.match(layer_name)]
            if not suitable_layers:
                raise ValueError('suitable layer for {} output is not found'.format(output_name))

            if len(suitable_layers) > 1:
                raise ValueError('more than 1 layers matched to regular expression, please specify more detailed regex')

            return suitable_layers[0]

        self.loc_out = find_layer(loc_out_regex, 'loc', raw_outputs)
        self.main_conf_out = find_layer(main_conf_out_regex, 'main confidence', raw_outputs)
        add_conf_with_bias = [layer_name + '/add_' for layer_name in self.add_conf_outs]
        if not contains_all(raw_outputs, self.add_conf_outs) and contains_all(raw_outputs, add_conf_with_bias):
            self.add_conf_outs = add_conf_with_bias

        self.outputs_verified = True
