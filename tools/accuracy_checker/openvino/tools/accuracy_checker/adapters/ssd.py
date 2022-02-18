"""
Copyright (c) 2018-2022 Intel Corporation

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

import itertools
import math
import re

import numpy as np

from .adapter import Adapter
from ..config import ConfigValidator, StringField, NumberField, ListField, BoolField
from ..postprocessor import NMS
from ..representation import DetectionPrediction, ContainerPrediction


class SSDAdapter(Adapter):
    """
    Class for converting output of SSD model to DetectionPrediction representation
    """
    __provider__ = 'ssd'
    prediction_types = (DetectionPrediction, )

    def configure(self):
        super().configure()
        self.outputs_verified = False

    def select_output_blob(self, outputs):
        super().select_output_blob(outputs)
        self.output_blob = self.check_output_name(self.output_blob, outputs)
        self.outputs_verified = True

    def process(self, raw, identifiers, frame_meta):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
        Returns:
            list of DetectionPrediction objects
        """
        prediction_batch = self._extract_predictions(raw, frame_meta)
        if not self.outputs_verified:
            self.select_output_blob(prediction_batch)
        prediction_batch = prediction_batch[self.output_blob]
        prediction_batch = prediction_batch.reshape(-1, 7)
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


class PyTorchSSDDecoder(Adapter):
    """
    Class for converting output of PyTorch SSD models to DetectionPrediction representation
    """
    __provider__ = 'pytorch_ssd_decoder'

    @classmethod
    def validate_config(cls, config, fetch_only=False, **kwargs):
        return super().validate_config(
            config, fetch_only=fetch_only, on_extra_argument=ConfigValidator.ERROR_ON_EXTRA_ARGUMENT
        )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'scores_out': StringField(description="Scores output layer name."),
            'boxes_out': StringField(description="Boxes output layer name."),
            'confidence_threshold': NumberField(optional=True, default=0.05, description="Confidence threshold."),
            'nms_threshold': NumberField(optional=True, default=0.5, description="NMS threshold."),
            'keep_top_k': NumberField(optional=True, value_type=int, default=200, description="Keep top K."),
            'feat_size': ListField(
                optional=True, description='Feature sizes list',
                value_type=ListField(value_type=NumberField(min_value=1, value_type=int))
            ),
            'do_softmax': BoolField(
                optional=True, default=True, description='Softmax operation should be applied to scores or not'
            )
        })

        return parameters

    def configure(self):
        self.scores_out = self.get_value_from_config('scores_out')
        self.boxes_out = self.get_value_from_config('boxes_out')
        self.outputs_verified = False
        self.confidence_threshold = self.get_value_from_config('confidence_threshold')
        self.nms_threshold = self.get_value_from_config('nms_threshold')
        self.keep_top_k = self.get_value_from_config('keep_top_k')
        self.do_softmax = self.get_value_from_config('do_softmax')
        feat_size = self.get_value_from_config('feat_size')

        # Set default values according to:
        # https://github.com/mlcommons/inference/tree/master/cloud/single_stage_detector
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.feat_size = [[50, 50], [25, 25], [13, 13], [7, 7], [3, 3], [3, 3]] if feat_size is None else feat_size
        self.scales = [21, 45, 99, 153, 207, 261, 315]
        self.strides = [3, 3, 2, 2, 2, 2]
        self.scale_xy = 0.1
        self.scale_wh = 0.2

    def select_output_blob(self, outputs):
        self.scores_out = self.check_output_name(self.scores_out, outputs)
        self.boxes_out = self.check_output_name(self.boxes_out, outputs)
        self.outputs_verified = True

    @staticmethod
    def softmax(x, axis=0):
        return np.transpose(np.transpose(np.exp(x)) * np.reciprocal(np.sum(np.exp(x), axis=axis)))

    @staticmethod
    def default_boxes(fig_size, feat_size, scales, aspect_ratios):

        fig_size_w, fig_size_h = fig_size
        scales = [(int(s * fig_size_w / 300), int(s * fig_size_h / 300)) for s in scales]
        fkw, fkh = np.transpose(feat_size)

        default_boxes = []
        for idx, sfeat in enumerate(feat_size):
            sfeat_w, sfeat_h = sfeat
            sk1 = scales[idx][0] / fig_size_w
            sk2 = scales[idx + 1][1] / fig_size_h
            sk3 = math.sqrt(sk1 * sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]
            for alpha in aspect_ratios[idx]:
                w, h = sk1 * math.sqrt(alpha), sk1 / math.sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat_w), range(sfeat_h)):
                    cx, cy = (j + 0.5) / fkh[idx], (i + 0.5) / fkw[idx]
                    default_boxes.append((cx, cy, w, h))
        default_boxes = np.clip(default_boxes, 0, 1)

        return default_boxes

    def process(self, raw, identifiers, frame_meta):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
        Returns:
            list of DetectionPrediction objects
        """

        raw_outputs = self._extract_predictions(raw, frame_meta)
        if not self.outputs_verified:
            self.select_output_blob(raw_outputs)

        batch_scores = raw_outputs[self.scores_out]
        batch_boxes = raw_outputs[self.boxes_out]
        need_transpose = np.shape(batch_boxes)[-1] != 4

        result = []
        for identifier, scores, boxes, meta in zip(identifiers, batch_scores, batch_boxes, frame_meta):
            detections = {'labels': [], 'scores': [], 'x_mins': [], 'y_mins': [], 'x_maxs': [], 'y_maxs': []}
            image_info = meta.get("image_info")[0:2]

            # Default boxes
            dboxes = self.default_boxes(image_info, self.feat_size, self.scales, self.aspect_ratios)

            # Scores
            scores = np.transpose(scores) if need_transpose else scores
            if self.do_softmax:
                scores = self.softmax(scores, axis=1)

            # Boxes
            boxes = np.transpose(boxes) if need_transpose else boxes
            boxes[:, :2] = self.scale_xy * boxes[:, :2]
            boxes[:, 2:] = self.scale_wh * boxes[:, 2:]
            boxes[:, :2] = boxes[:, :2] * dboxes[:, 2:] + dboxes[:, :2]
            boxes[:, 2:] = np.exp(boxes[:, 2:]) * dboxes[:, 2:]

            for label, score in enumerate(np.transpose(scores)):

                # Skip background label
                if label == 0:
                    continue

                # Filter out detections with score < confidence_threshold
                mask = score > self.confidence_threshold
                filtered_boxes, filtered_score = boxes[mask, :], score[mask]
                if filtered_score.size == 0:
                    continue

                # Transform to format (x_min, y_min, x_max, y_max)
                x_mins = (filtered_boxes[:, 0] - 0.5 * filtered_boxes[:, 2])
                y_mins = (filtered_boxes[:, 1] - 0.5 * filtered_boxes[:, 3])
                x_maxs = (filtered_boxes[:, 0] + 0.5 * filtered_boxes[:, 2])
                y_maxs = (filtered_boxes[:, 1] + 0.5 * filtered_boxes[:, 3])

                # Apply NMS
                keep = NMS.nms(x_mins, y_mins, x_maxs, y_maxs, filtered_score, self.nms_threshold,
                               include_boundaries=False, keep_top_k=self.keep_top_k)

                filtered_score = filtered_score[keep]
                x_mins = x_mins[keep]
                y_mins = y_mins[keep]
                x_maxs = x_maxs[keep]
                y_maxs = y_maxs[keep]

                # Keep topK
                # Applied just after NMS - no additional sorting is required for filtered_score array
                filtered_score = filtered_score[:self.keep_top_k]
                x_mins = x_mins[:self.keep_top_k]
                y_mins = y_mins[:self.keep_top_k]
                x_maxs = x_maxs[:self.keep_top_k]
                y_maxs = y_maxs[:self.keep_top_k]

                # Save detections
                labels = np.full_like(filtered_score, label)
                detections['labels'].extend(labels)
                detections['scores'].extend(filtered_score)
                detections['x_mins'].extend(x_mins)
                detections['y_mins'].extend(y_mins)
                detections['x_maxs'].extend(x_maxs)
                detections['y_maxs'].extend(y_maxs)

            result.append(
                DetectionPrediction(
                    identifier, detections['labels'], detections['scores'], detections['x_mins'],
                    detections['y_mins'], detections['x_maxs'], detections['y_maxs']
                )
            )

        return result


class FacePersonAdapter(Adapter):
    __provider__ = 'face_person_detection'
    prediction_types = (DetectionPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'face_out': StringField(description="Face detection output layer name."),
            'person_out': StringField(description="Person detection output layer name"),
        })

        return parameters

    @classmethod
    def validate_config(cls, config, fetch_only=False, **kwargs):
        return super().validate_config(
            config, fetch_only=fetch_only, on_extra_argument=ConfigValidator.ERROR_ON_EXTRA_ARGUMENT
        )

    def configure(self):
        self.face_detection_out = self.launcher_config['face_out']
        self.person_detection_out = self.launcher_config['person_out']
        self.face_adapter = SSDAdapter(self.launcher_config, self.label_map, self.face_detection_out)
        self.person_adapter = SSDAdapter(self.launcher_config, self.label_map, self.person_detection_out)

    def process(self, raw, identifiers, frame_meta):
        face_batch_result = self.face_adapter.process(raw, identifiers, frame_meta)
        person_batch_result = self.person_adapter.process(raw, identifiers, frame_meta)
        result = [ContainerPrediction({self.face_detection_out: face_result, self.person_detection_out: person_result})
                  for face_result, person_result in zip(face_batch_result, person_batch_result)]

        return result


class SSDAdapterMxNet(Adapter):
    """
    Class for converting output of MXNet SSD model to DetectionPrediction representation
    """
    __provider__ = 'ssd_mxnet'

    def process(self, raw, identifiers, frame_meta):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model which is ndarray of shape (batch, det_count, 6),
                 each detection is defined by 6 values: class_id, prob, x_min, y_min, x_max, y_max
        Returns:
            list of DetectionPrediction objects
        """
        raw_outputs = self._extract_predictions(raw, frame_meta)
        self.select_output_blob(raw_outputs)
        result = []
        for identifier, prediction_batch in zip(identifiers, raw_outputs[self.output_blob]):
            # Filter detections (get only detections with class_id >= 0)
            detections = prediction_batch[np.where(prediction_batch[:, 0] >= 0)]
            # Append detections to results
            result.append(DetectionPrediction(identifier, *zip(*detections)))

        return result


class SSDONNXAdapter(Adapter):
    __provider__ = 'ssd_onnx'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update(
            {
                'labels_out': StringField(description='name (or regex for it) of output layer with labels'),
                'scores_out': StringField(
                    description='name (or regex for it) of output layer with scores', optional=True
                ),
                'bboxes_out': StringField(description='name (or regex for it) of output layer with bboxes')
            }
        )
        return parameters

    def configure(self):
        self.labels_out = self.get_value_from_config('labels_out')
        self.scores_out = self.get_value_from_config('scores_out')
        self.bboxes_out = self.get_value_from_config('bboxes_out')
        self.outputs_verified = False

    def process(self, raw, identifiers, frame_meta):
        raw_outputs = self._extract_predictions(raw, frame_meta)
        results = []
        if not self.outputs_verified:
            self._get_output_names(raw_outputs)
        boxes_out, labels_out = raw_outputs[self.bboxes_out], raw_outputs[self.labels_out]
        if len(boxes_out.shape) == 2:
            boxes_out = np.expand_dims(boxes_out, 0)
            labels_out = np.expand_dims(labels_out, 0)
        for idx, (identifier, bboxes, labels) in enumerate(zip(
                identifiers, boxes_out, labels_out
        )):
            if self.scores_out:
                scores = raw_outputs[self.scores_out][idx]
                x_mins, y_mins, x_maxs, y_maxs = bboxes.T
            else:
                x_mins, y_mins, x_maxs, y_maxs, scores = bboxes.T
            if labels.ndim > 1:
                labels = np.squeeze(labels)
            if scores.ndim > 1:
                scores = np.squeeze(scores)
            if x_mins.ndim > 1:
                x_mins = np.squeeze(x_mins)
                y_mins = np.squeeze(y_mins)
                x_maxs = np.squeeze(x_maxs)
                y_maxs = np.squeeze(y_maxs)
            results.append(
                DetectionPrediction(
                    identifier, labels, scores, x_mins, y_mins, x_maxs, y_maxs))

        return results

    def _get_output_names(self, raw_outputs):
        labels_regex = re.compile(self.labels_out)
        scores_regex = re.compile(self.scores_out) if self.scores_out else ''
        bboxes_regex = re.compile(self.bboxes_out)

        def find_layer(regex, output_name, all_outputs):
            suitable_layers = [layer_name for layer_name in all_outputs if regex.match(layer_name)]
            if not suitable_layers:
                raise ValueError('suitable layer for {} output is not found'.format(output_name))

            if len(suitable_layers) > 1:
                raise ValueError('more than 1 layers matched to regular expression, please specify more detailed regex')

            return suitable_layers[0]

        self.labels_out = find_layer(labels_regex, 'labels', raw_outputs)
        self.scores_out = find_layer(scores_regex, 'scores', raw_outputs) if self.scores_out else None
        self.bboxes_out = find_layer(bboxes_regex, 'bboxes', raw_outputs)

        self.outputs_verified = True


class SSDMultiLabelAdapter(Adapter):
    __provider__ = 'ssd_multilabel'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'scores_out': StringField(description='scores output'),
            'boxes_out': StringField(description='boxes output'),
            'confidence_threshold': NumberField(optional=True, default=0.01, description="confidence threshold"),
            'nms_threshold': NumberField(optional=True, default=0.45, description="NMS threshold"),
            'keep_top_k': NumberField(optional=True, value_type=int, default=200, description="keep top K during NMS"),
            'diff_coord_order': BoolField(optional=True, default=False,
                                          description='Ordering convention of coordinates differs from standard'),
            'max_detections': NumberField(optional=True, value_type=int, description="maximal number of detections")
        })
        return params

    def configure(self):
        self.scores_out = self.get_value_from_config('scores_out')
        self.boxes_out = self.get_value_from_config('boxes_out')
        self.confidence_threshold = self.get_value_from_config('confidence_threshold')
        self.iou_threshold = self.get_value_from_config('nms_threshold')
        self.keep_top_k = self.get_value_from_config('keep_top_k')
        self.diff_coord_order = self.get_value_from_config('diff_coord_order')
        self.max_detections = self.get_value_from_config('max_detections')
        self.outputs_verified = False

    def select_output_blob(self, outputs):
        self.scores_out = self.check_output_name(self.scores_out, outputs)
        self.boxes_out = self.check_output_name(self.boxes_out, outputs)
        self.outputs_verified = True

    def process(self, raw, identifiers, frame_meta):
        result = []
        raw_output = self._extract_predictions(raw, frame_meta)
        if not self.outputs_verified:
            self.select_output_blob(raw_output)

        for identifier, logits, boxes in zip(identifiers, raw_output[self.scores_out], raw_output[self.boxes_out]):
            detections = {'labels': [], 'scores': [], 'x_mins': [], 'y_mins': [], 'x_maxs': [], 'y_maxs': []}
            for class_index in range(1, logits.shape[-1]):
                probs = logits[:, class_index]
                mask = probs > self.confidence_threshold
                probs = probs[mask]
                if probs.size == 0:
                    continue
                subset_boxes = boxes[mask, :]

                if self.diff_coord_order:
                    y_mins, x_mins, y_maxs, x_maxs = subset_boxes.T
                else:
                    x_mins, y_mins, x_maxs, y_maxs = subset_boxes.T

                keep = NMS.nms(x_mins, y_mins, x_maxs, y_maxs, probs, self.iou_threshold, include_boundaries=False,
                               keep_top_k=self.keep_top_k)

                filtered_probs = probs[keep]
                x_mins = x_mins[keep]
                y_mins = y_mins[keep]
                x_maxs = x_maxs[keep]
                y_maxs = y_maxs[keep]

                labels = [class_index] * filtered_probs.size
                detections['labels'].extend(labels)
                detections['scores'].extend(filtered_probs)
                detections['x_mins'].extend(x_mins)
                detections['y_mins'].extend(y_mins)
                detections['x_maxs'].extend(x_maxs)
                detections['y_maxs'].extend(y_maxs)

            if self.max_detections:
                max_ids = np.argsort(-np.array(detections['scores']))[:self.max_detections]
                detections['labels'] = np.array(detections['labels'])[max_ids]
                detections['scores'] = np.array(detections['scores'])[max_ids]
                detections['x_mins'] = np.array(detections['x_mins'])[max_ids]
                detections['y_mins'] = np.array(detections['y_mins'])[max_ids]
                detections['x_maxs'] = np.array(detections['x_maxs'])[max_ids]
                detections['y_maxs'] = np.array(detections['y_maxs'])[max_ids]

            result.append(DetectionPrediction(identifier, detections['labels'], detections['scores'],
                                              detections['x_mins'], detections['y_mins'], detections['x_maxs'],
                                              detections['y_maxs']))
        return result
