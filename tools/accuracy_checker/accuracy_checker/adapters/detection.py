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

import itertools
import math
import warnings
from collections import namedtuple
import numpy as np

from ..adapters import Adapter
from ..config import ConfigValidator, BaseField, NumberField, StringField, ListField, ConfigError, BoolField
from ..postprocessor.nms import NMS
from ..representation import DetectionPrediction
from ..utils import get_or_parse_value, softmax

FaceDetectionLayerOutput = namedtuple('FaceDetectionLayerOutput', ['prob_name', 'reg_name', 'anchor_index',
    'anchor_size', 'win_scale', 'win_length', 'win_trans_x', 'win_trans_y'])


class TFObjectDetectionAPIAdapter(Adapter):
    """
    Class for converting output of SSD model to DetectionPrediction representation
    """
    __provider__ = 'tf_object_detection'

    @classmethod
    def validate_config(cls, config, fetch_only=False, **kwargs):
        return super().validate_config(
            config, fetch_only=fetch_only, on_extra_argument=ConfigValidator.ERROR_ON_EXTRA_ARGUMENT
        )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'classes_out': StringField(description="Classes output layer name."),
            'boxes_out': StringField(description="Boxes output layer name."),
            'scores_out': StringField(description="Scores output layer name."),
            'num_detections_out': StringField(description="Number of detections output layer name.")
        })
        return parameters

    def configure(self):
        self.classes_out = self.get_value_from_config('classes_out')
        self.boxes_out = self.get_value_from_config('boxes_out')
        self.scores_out = self.get_value_from_config('scores_out')
        self.num_detections_out = self.get_value_from_config('num_detections_out')
        self.outputs_verified = False

    def select_output_blob(self, outputs):
        self.classes_out = self.check_output_name(self.classes_out, outputs)
        self.boxes_out = self.check_output_name(self.boxes_out, outputs)
        self.scores_out = self.check_output_name(self.scores_out, outputs)
        self.num_detections_out = self.check_output_name(self.num_detections_out, outputs)
        self.outputs_verified = True

    def process(self, raw, identifiers=None, frame_meta=None):
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
        classes_batch = prediction_batch[self.classes_out]
        scores_batch = prediction_batch[self.scores_out]
        boxes_batch = prediction_batch[self.boxes_out]
        num_detections_batch = prediction_batch[self.num_detections_out].astype(int)

        result = []
        for identifier, classes, scores, boxes, num_detections in zip(
                identifiers, classes_batch, scores_batch, boxes_batch, num_detections_batch
        ):
            valid_classes = classes[:num_detections]
            valid_scores = scores[:num_detections]
            valid_boxes = boxes[:num_detections]
            y_mins, x_mins, y_maxs, x_maxs = valid_boxes.T
            result.append(DetectionPrediction(identifier, valid_classes, valid_scores, x_mins, y_mins, x_maxs, y_maxs))

        return result


class ClassAgnosticDetectionAdapter(Adapter):
    """
    Class for converting 'boxes' [n,5] output of detection model to
    DetectionPrediction representation
    """
    __provider__ = 'class_agnostic_detection'
    prediction_types = (DetectionPrediction,)

    @classmethod
    def validate_config(cls, config, fetch_only=False, **kwargs):
        return super().validate_config(
            config, fetch_only=fetch_only, on_extra_argument=ConfigValidator.ERROR_ON_EXTRA_ARGUMENT
        )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'output_blob': StringField(optional=True, default=None, description="Output blob name."),
            'scale': BaseField(optional=True, default=1.0, description="Scale factor for bboxes."),
        })
        return parameters

    def configure(self):
        self.out_blob_name = self.get_value_from_config('output_blob')
        self.scale = get_or_parse_value(self.get_value_from_config('scale'))
        self.output_verified = False
        if isinstance(self.scale, list):
            self.scale = self.scale * 2

    def select_output_blob(self, outputs):
        self.output_verified = True
        if self.out_blob_name:
            self.out_blob_name = self.check_output_name(self.out_blob_name, outputs)
            return
        self.out_blob_name = self._find_output(outputs)
        return

    def process(self, raw, identifiers, frame_meta):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
            frame_meta: image metadata
        Returns:
            list of DetectionPrediction objects
        """
        predictions = self._extract_predictions(raw, frame_meta)
        if not self.output_verified:
            self.select_output_blob(predictions)
        prediction_batch = predictions[self.out_blob_name]
        if np.ndim(prediction_batch) == 2 and len(identifiers) == 1:
            prediction_batch = np.expand_dims(prediction_batch, 0)

        result = []
        for identifier, pred in zip(identifiers, prediction_batch):
            prediction_mask = np.where(pred[:, -1] > 0.0)
            valid_detections = pred[prediction_mask]

            bboxes = self.scale * valid_detections[:, :-1]
            scores = valid_detections[:, -1]
            labels = np.ones([len(scores)], dtype=np.int32)

            result.append(DetectionPrediction(identifier, labels, scores, *zip(*bboxes)))

        return result

    @staticmethod
    def _find_output(predictions):
        filter_outputs = [
            output_name for output_name, out_data in predictions.items()
            if len(np.shape(out_data)) in [2, 3] and np.shape(out_data)[-1] == 5
        ]
        if not filter_outputs:
            raise ConfigError('Suitable output layer not found')
        if len(filter_outputs) > 1:
            warnings.warn(
                'There is several suitable outputs {}. The first will be used. '.format(', '.join(filter_outputs)) +
                'If you need to use another layer, please specify it explicitly'
            )
        return filter_outputs[0]


class RFCNCaffe(Adapter):
    __provider__ = 'rfcn_class_agnostic'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'cls_out': StringField(description='bboxes predicted classes score out'),
            'bbox_out': StringField(description='bboxes output with shape [N, 8]'),
            'rois_out': StringField(description='rois features output')
        })
        return params

    def configure(self):
        self.cls_out = self.get_value_from_config('cls_out')
        self.bbox_out = self.get_value_from_config('bbox_out')
        self.rois_out = self.get_value_from_config('rois_out')
        self.outputs_verified = False

    def select_output_blob(self, outputs):
        self.cls_out = self.check_output_name(self.cls_out, outputs)
        self.bbox_out = self.check_output_name(self.bbox_out, outputs)
        self.rois_out = self.check_output_name(self.rois_out, outputs)
        self.outputs_verified = True

    def get_proposals(self, raw_out):
        predicted_proposals = raw_out.get(self.rois_out)
        if predicted_proposals is None:
            if self.rois_out + '.0' in raw_out:
                predicted_proposals = raw_out[self.rois_out + '.0']
            else:
                raise ConfigError("output blobs do not contain {}".format(self.rois_out))
        return predicted_proposals

    @staticmethod
    def get_scale(meta):
        if 'scale_x' in meta:
            x_scale = meta['scale_x']
            y_scale = meta['scale_y']
            return x_scale, y_scale
        original_image_size = meta['image_size'][:2]
        image_input = [shape for shape in meta['input_shape'].values() if len(shape) == 4]
        assert image_input, "image input not found"
        assert len(image_input) == 1, 'several input images detected'
        image_input = image_input[0]
        if image_input[1] == 3:
            processed_image_size = image_input[2:]
        else:
            processed_image_size = image_input[1:3]
        y_scale = processed_image_size[0] / original_image_size[0]
        x_scale = processed_image_size[1] / original_image_size[1]
        return x_scale, y_scale

    def process(self, raw, identifiers, frame_meta):
        assert len(identifiers) == 1, '{} adapter support only batch size 1'.format(self.__provider__)
        raw_out = self._extract_predictions(raw, frame_meta)
        if not self.outputs_verified:
            self.select_output_blob(raw_out)
        predicted_classes = raw_out[self.cls_out]
        predicted_deltas = raw_out[self.bbox_out]
        predicted_proposals = self.get_proposals(raw_out)
        x_scale, y_scale = self.get_scale(frame_meta[0])
        real_det_num = np.argwhere(predicted_proposals[:, 0] == -1)
        if np.size(real_det_num) != 0:
            real_det_num = real_det_num[0, 0]
            predicted_proposals = predicted_proposals[:real_det_num]
            predicted_deltas = predicted_deltas[:real_det_num]
            predicted_classes = predicted_classes[:real_det_num]
        predicted_proposals[:, 1::2] /= x_scale
        predicted_proposals[:, 2::2] /= y_scale
        assert len(predicted_classes.shape) == 2
        assert predicted_deltas.shape[-1] == 8
        predicted_boxes = self.bbox_transform_inv(predicted_proposals, predicted_deltas)
        num_classes = predicted_classes.shape[-1] - 1  # skip background
        x_mins, y_mins, x_maxs, y_maxs = predicted_boxes[:, 4:].T
        detections = {'labels': [], 'scores': [], 'x_mins': [], 'y_mins': [], 'x_maxs': [], 'y_maxs': []}
        for cls_id in range(num_classes):
            cls_scores = predicted_classes[:, cls_id + 1]
            keep = NMS.nms(x_mins, y_mins, x_maxs, y_maxs, cls_scores, 0.3, include_boundaries=False)
            filtered_score = cls_scores[keep]
            x_cls_mins, y_cls_mins, x_cls_maxs, y_cls_maxs = x_mins[keep], y_mins[keep], x_maxs[keep], y_maxs[keep]
            # Save detections
            labels = np.full_like(filtered_score, cls_id + 1)
            detections['labels'].extend(labels)
            detections['scores'].extend(filtered_score)
            detections['x_mins'].extend(x_cls_mins)
            detections['y_mins'].extend(y_cls_mins)
            detections['x_maxs'].extend(x_cls_maxs)
            detections['y_maxs'].extend(y_cls_maxs)
        return [DetectionPrediction(
            identifiers[0], detections['labels'], detections['scores'], detections['x_mins'],
            detections['y_mins'], detections['x_maxs'], detections['y_maxs']
        )]

    @staticmethod
    def bbox_transform_inv(boxes, deltas):
        if boxes.shape[0] == 0:
            return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)
        boxes = boxes.astype(deltas.dtype, copy=False)
        widths = boxes[:, 3] - boxes[:, 1] + 1.0
        heights = boxes[:, 4] - boxes[:, 2] + 1.0
        ctr_x = boxes[:, 1] + 0.5 * widths
        ctr_y = boxes[:, 2] + 0.5 * heights
        dx = deltas[:, 0::4]
        dy = deltas[:, 1::4]
        dw = deltas[:, 2::4]
        dh = deltas[:, 3::4]
        pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
        pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
        pred_w = np.exp(dw) * widths[:, np.newaxis]
        pred_h = np.exp(dh) * heights[:, np.newaxis]
        pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes


class FaceBoxesAdapter(Adapter):
    """
    Class for converting output of FaceBoxes models to DetectionPrediction representation
    """
    __provider__ = 'faceboxes'

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
        })
        return parameters

    def configure(self):
        self.scores_out = self.get_value_from_config('scores_out')
        self.boxes_out = self.get_value_from_config('boxes_out')
        self.outputs_verified = False
        self._anchors_cache = {}

        # Set default values
        self.min_sizes = [[32, 64, 128], [256], [512]]
        self.steps = [32, 64, 128]
        self.variance = [0.1, 0.2]
        self.confidence_threshold = 0.05
        self.nms_threshold = 0.3
        self.keep_top_k = 750

    def select_output_blob(self, outputs):
        self.scores_out = self.check_output_name(self.scores_out, outputs)
        self.boxes_out = self.check_output_name(self.boxes_out, outputs)
        self.outputs_verified = True

    @staticmethod
    def calculate_anchors(list_x, list_y, min_size, image_size, step):
        anchors = []
        s_kx = min_size / image_size[1]
        s_ky = min_size / image_size[0]
        dense_cx = [x * step / image_size[1] for x in list_x]
        dense_cy = [y * step / image_size[0] for y in list_y]
        for cy, cx in itertools.product(dense_cy, dense_cx):
            anchors.append([cx, cy, s_kx, s_ky])
        return anchors

    def calculate_anchors_zero_level(self, f_x, f_y, min_sizes, image_size, step):
        anchors = []
        for min_size in min_sizes:
            if min_size == 32:
                list_x = [f_x + 0, f_x + 0.25, f_x + 0.5, f_x + 0.75]
                list_y = [f_y + 0, f_y + 0.25, f_y + 0.5, f_y + 0.75]
            elif min_size == 64:
                list_x = [f_x + 0, f_x + 0.5]
                list_y = [f_y + 0, f_y + 0.5]
            else:
                list_x = [f_x + 0.5]
                list_y = [f_y + 0.5]
            anchors.extend(self.calculate_anchors(list_x, list_y, min_size, image_size, step))
        return anchors

    def prior_boxes(self, feature_maps, image_size):
        anchors = []
        for k, f in enumerate(feature_maps):
            for i, j in itertools.product(range(f[0]), range(f[1])):
                if k == 0:
                    anchors.extend(self.calculate_anchors_zero_level(j, i, self.min_sizes[k],
                                                                     image_size, self.steps[k]))
                else:
                    anchors.extend(self.calculate_anchors([j + 0.5], [i + 0.5], self.min_sizes[k][0],
                                                          image_size, self.steps[k]))
        anchors = np.clip(anchors, 0, 1)

        return anchors

    def process(self, raw, identifiers, frame_meta):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
            frame_meta: images metadata
        Returns:
            list of DetectionPrediction objects
        """
        raw_outputs = self._extract_predictions(raw, frame_meta)
        if not self.outputs_verified:
            self.select_output_blob(raw_outputs)

        batch_scores = raw_outputs[self.scores_out]
        batch_boxes = raw_outputs[self.boxes_out]

        result = []
        for identifier, scores, boxes, meta in zip(identifiers, batch_scores, batch_boxes, frame_meta):
            detections = {'labels': [], 'scores': [], 'x_mins': [], 'y_mins': [], 'x_maxs': [], 'y_maxs': []}
            image_info = meta.get("image_info")[0:2]

            # Prior boxes
            if (image_info[0], image_info[1]) not in self._anchors_cache:
                feature_maps = [[math.ceil(image_info[0] / step), math.ceil(image_info[1] / step)] for step in
                                self.steps]
                prior_data = self.prior_boxes(feature_maps, image_info)
                self._anchors_cache[(image_info[0], image_info[1])] = prior_data
            else:
                prior_data = self._anchors_cache[(image_info[0], image_info[1])]

            # Boxes
            boxes[:, :2] = self.variance[0] * boxes[:, :2]
            boxes[:, 2:] = self.variance[1] * boxes[:, 2:]
            boxes[:, :2] = boxes[:, :2] * prior_data[:, 2:] + prior_data[:, :2]
            boxes[:, 2:] = np.exp(boxes[:, 2:]) * prior_data[:, 2:]

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
                x_mins, y_mins, x_maxs, y_maxs = x_mins[keep], y_mins[keep], x_maxs[keep], y_maxs[keep]

                # Keep topK
                # Applied just after NMS - no additional sorting is required for filtered_score array
                if filtered_score.size > self.keep_top_k:
                    filtered_score = filtered_score[:self.keep_top_k]
                    x_mins = x_mins[:self.keep_top_k]
                    y_mins = y_mins[:self.keep_top_k]
                    x_maxs = x_maxs[:self.keep_top_k]
                    y_maxs = y_maxs[:self.keep_top_k]

                # Save detections
                labels = np.full_like(filtered_score, label, dtype=int)
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


class FaceDetectionAdapter(Adapter):
    """
    Class for converting output of Face Detection model to DetectionPrediction representation
    """
    __provider__ = 'face_detection'
    predcition_types = (DetectionPrediction,)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'score_threshold': NumberField(value_type=float, min_value=0, max_value=1, default=0.35, optional=True,
                                           description='Score threshold value used to discern whether a face is valid'),
            'layer_names': ListField(value_type=str, optional=False, description='Target output layer base names'),
            'anchor_sizes': ListField(value_type=int, optional=False,
                                      description='Anchor sizes for each base output layer'),
            'window_scales': ListField(value_type=int, optional=False,
                                       description='Window scales for each base output layer'),
            'window_lengths': ListField(value_type=int, optional=False,
                                        description='Window lengths for each base output layer'),
        })
        return parameters

    def configure(self):
        self.score_threshold = self.get_value_from_config('score_threshold')
        self.layer_info = {
            'layer_names': self.get_value_from_config('layer_names'),
            'anchor_sizes': self.get_value_from_config('anchor_sizes'),
            'window_scales': self.get_value_from_config('window_scales'),
            'window_lengths': self.get_value_from_config('window_lengths')
        }
        self.outputs_verified = False
        if len({len(x) for x in self.layer_info.values()}) != 1:
            raise ConfigError('There must be equal number of layer names, anchor sizes, '
                              'window scales, and window sizes')
        self.output_layers = self.generate_output_layer_info()

    def generate_output_layer_info(self):
        """
        Generates face detection layer information,
        which is referenced in process function
        """
        output_layers = []

        for i in range(len(self.layer_info['layer_names'])):
            start = 1.5
            anchor_size = self.layer_info['anchor_sizes'][i]
            layer_name = self.layer_info['layer_names'][i]
            window_scale = self.layer_info['window_scales'][i]
            window_length = self.layer_info['window_lengths'][i]
            if anchor_size % 3 == 0:
                start = -anchor_size / 3.0
            elif anchor_size % 2 == 0:
                start = -anchor_size / 2.0 + 0.5
            k = 1
            for row in range(anchor_size):
                for col in range(anchor_size):
                    out_layer = FaceDetectionLayerOutput(
                        prob_name=layer_name + '/prob',
                        reg_name=layer_name + '/bb',
                        anchor_index=k - 1,
                        anchor_size=anchor_size * anchor_size,
                        win_scale=window_scale,
                        win_length=window_length,
                        win_trans_x=float((start + col) / anchor_size),
                        win_trans_y=float((start + row) / anchor_size)
                    )
                    output_layers.append(out_layer)
                    k += 1
        return output_layers

    def select_output_blob(self, outputs):
        updated_outputs = []
        for out_name in self.output_layers:
            updated_outputs.append(self.check_output_name(out_name, outputs))
        self.output_layers = updated_outputs

    def process(self, raw, identifiers, frame_meta):
        result = []
        if not self.outputs_verified:
            self.select_output_blob(raw)
        for batch_index, identifier in enumerate(identifiers):
            detections = {'labels': [], 'scores': [], 'x_mins': [], 'y_mins': [], 'x_maxs': [], 'y_maxs': []}
            scale_factor = frame_meta[batch_index]['scales'][0]
            for layer in self.output_layers:
                prob_arr = raw[batch_index][layer.prob_name]
                prob_dims = raw[batch_index][layer.prob_name].shape
                reg_arr = raw[batch_index][layer.reg_name]

                output_height = prob_dims[2]
                output_width = prob_dims[3]

                anchor_loc = layer.anchor_size + layer.anchor_index
                prob_data = prob_arr[0][anchor_loc]

                for row in range(output_height):
                    for col in range(output_width):
                        score = prob_data[row][col]
                        if score >= self.score_threshold:
                            candidate_x = (col + layer.win_trans_x) * layer.win_scale - 0.5
                            candidate_y = (row + layer.win_trans_y) * layer.win_scale - 0.5
                            candidate_width = layer.win_length
                            candidate_height = layer.win_length

                            reg_x = reg_arr[0][layer.anchor_index * 4 + 0][row][col] * layer.win_length
                            reg_y = reg_arr[0][layer.anchor_index * 4 + 1][row][col] * layer.win_length
                            reg_width = reg_arr[0][layer.anchor_index * 4 + 2][row][col] * layer.win_length
                            reg_height = reg_arr[0][layer.anchor_index * 4 + 3][row][col] * layer.win_length

                            candidate_x += reg_x
                            candidate_y += reg_y
                            candidate_width += reg_width
                            candidate_height += reg_height

                            min_x = scale_factor * (candidate_x) + 0.5
                            min_y = scale_factor * (candidate_y) + 0.5
                            width = scale_factor * candidate_width + 0.5
                            height = scale_factor * candidate_height + 0.5

                            detections['x_mins'].append(min_x)
                            detections['y_mins'].append(min_y)
                            detections['x_maxs'].append(min_x + width)
                            detections['y_maxs'].append(min_y + height)
                            detections['scores'].append(score)

            result.append(
                DetectionPrediction(identifier=identifier, labels=np.zeros_like(detections['scores']),
                                    x_mins=detections['x_mins'], y_mins=detections['y_mins'],
                                    x_maxs=detections['x_maxs'], y_maxs=detections['y_maxs'],
                                    scores=detections['scores'])
            )

        return result


class FaceDetectionRefinementAdapter(Adapter):
    __provider__ = 'face_detection_refinement'
    prediction_types = (DetectionPrediction,)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'threshold': NumberField(value_type=float, min_value=0, default=0.5, optional=False,
                                     description='Score threshold to determine as valid face candidate')
        })
        return parameters

    def configure(self):
        self.threshold = self.get_value_from_config('threshold')

    def process(self, raw, identifiers, frame_meta):
        if isinstance(raw, dict):
            candidates = frame_meta[0]['candidates']
            result = self.refine_candidates(candidates.identifier, [raw], candidates, self.threshold)
        else:
            candidates = frame_meta[0]['candidates']
            result = self.refine_candidates(identifiers[0], raw, candidates, self.threshold)

        return result

    @staticmethod
    def refine_candidates(identifier, raw, candidates, threshold):
        prob_name = 'prob_fd'
        reg_name = 'fc_bb'
        detections = {'scores': [], 'x_mins': [], 'y_mins': [], 'x_maxs': [], 'y_maxs': []}

        for i, prediction in enumerate(raw):
            prob_arr = prediction[prob_name]
            reg_arr = prediction[reg_name]
            score = prob_arr[0, 1, 0, 0]

            if score < threshold:
                continue

            width = candidates.x_maxs[i] - candidates.x_mins[i]
            height = candidates.y_maxs[i] - candidates.y_mins[i]

            center_x = candidates.x_mins[i] + (width - 1) / 2
            center_y = candidates.y_mins[i] + (height - 1) / 2

            regression = reg_arr[0, 0, 0, 0]

            reg_x = regression * width
            reg_y = reg_arr[0, 1, 0, 0] * height
            reg_width = reg_arr[0, 2, 0, 0] * width
            reg_height = reg_arr[0, 3, 0, 0] * height

            width += reg_width
            height += reg_height
            x = (center_x + reg_x) - width / 2.0 + 0.5
            y = (center_y + reg_y) - height / 2.0 + 0.5
            width += 0.5
            height += 0.5
            detections['scores'].append(score)
            detections['x_mins'].append(x)
            detections['y_mins'].append(y)
            detections['x_maxs'].append(x + width)
            detections['y_maxs'].append(y + height)

        return [
            DetectionPrediction(identifier=identifier, x_mins=detections['x_mins'], y_mins=detections['y_mins'],
                                x_maxs=detections['x_maxs'], y_maxs=detections['y_maxs'], scores=detections['scores'])
        ]


class FasterRCNNONNX(Adapter):
    __provider__ = 'faster_rcnn_onnx'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'labels_out': StringField(description='name of output layer with labels', optional=True),
            'scores_out': StringField(description='name of output layer with scores', optional=True),
            'boxes_out': StringField(description='name of output layer with bboxes')
        })
        return parameters

    def configure(self):
        self.boxes_out = self.get_value_from_config('boxes_out')
        self.labels_out = self.get_value_from_config('labels_out')
        self.scores_out = self.get_value_from_config('scores_out')
        if self.scores_out and not self.labels_out:
            raise ConfigError('all three outputs or bixrs_out and labels_out or only boxes_out should be provided')
        self.outputs_verified = False

    def select_output_blob(self, outputs):
        self.boxes_out = self.check_output_name(self.boxes_out, outputs)
        if self.scores_out:
            self.scores_out = self.check_output_name(self.scores_out, outputs)
        if self.labels_out:
            self.labels_out = self.check_output_name(self.labels_out, outputs)
        self.outputs_verified = True

    def process(self, raw, identifiers=None, frame_meta=None):
        raw_outputs = self._extract_predictions(raw, frame_meta)
        if not self.outputs_verified:
            self.select_output_blob(raw_outputs)
        identifier = identifiers[0]
        boxes = raw_outputs[self.boxes_out][:, :4]
        scores = raw_outputs[self.scores_out] if self.scores_out is not None else raw_outputs[self.boxes_out][:, 4]
        labels = raw_outputs[self.labels_out] if self.labels_out is not None else raw_outputs[self.boxes_out][:, 5]
        meta = frame_meta[0]
        im_scale_x = meta['scale_x']
        im_scale_y = meta['scale_y']
        boxes[:, 0::2] /= im_scale_x
        boxes[:, 1::2] /= im_scale_y
        x_mins, y_mins, x_maxs, y_maxs = boxes.T
        return [DetectionPrediction(identifier, labels, scores, x_mins, y_mins, x_maxs, y_maxs)]


class TwoStageDetector(Adapter):
    __provider__ = 'two_stage_detection'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'boxes_out': StringField(description='boxes output'),
            'cls_out': StringField(description='classes confidence output')
        })
        return params

    def configure(self):
        self.boxes_out = self.get_value_from_config('boxes_out')
        self.cls_out = self.get_value_from_config('cls_out')

    def process(self, raw, identifiers, frame_meta):
        raw_output = self._extract_predictions(raw, frame_meta)
        boxes_outputs = raw_output[self.boxes_out]
        if len(boxes_outputs.shape) == 2:
            boxes_outputs = np.expand_dims(boxes_outputs, 0)
        conf_outputs = raw_output[self.cls_out]
        if len(conf_outputs.shape) == 2:
            conf_outputs = np.expand_dims(conf_outputs, 0)
        result = []
        for identifier, boxes, conf in zip(identifiers, boxes_outputs, conf_outputs):
            x_mins, y_mins, w, h = boxes.T
            labels = np.argmax(conf, axis=1)
            scores = np.max(conf, axis=1)
            result.append(DetectionPrediction(identifier, labels, scores, x_mins, y_mins, x_mins + w, y_mins + h))
        return result


class DETRAdapter(Adapter):
    __provider__ = 'detr'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'scores_out': StringField(description='scores output'),
            'boxes_out': StringField(description='boxes output')
        })
        return params

    def configure(self):
        self.scores_out = self.get_value_from_config('scores_out')
        self.boxes_out = self.get_value_from_config('boxes_out')
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

        def box_cxcywh_to_xyxy(x):
            x_c, y_c, w, h = x.T
            b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
                 (x_c + 0.5 * w), (y_c + 0.5 * h)]
            return b

        def softmax_logits(logits):
            res = [np.exp(logit) / np.sum(np.exp(logit)) for logit in logits]
            return np.array(res)

        for identifier, logits, boxes in zip(identifiers, raw_output[self.scores_out], raw_output[self.boxes_out]):
            x_mins, y_mins, x_maxs, y_maxs = box_cxcywh_to_xyxy(boxes)
            scores = softmax_logits(logits)
            labels = np.argmax(scores[:, :-1], axis=-1)
            det_scores = np.max(scores[:, :-1], axis=-1)
            result.append(DetectionPrediction(identifier, labels, det_scores, x_mins, y_mins, x_maxs, y_maxs))

        return result


class UltraLightweightFaceDetectionAdapter(Adapter):
    """
    Class for converting output of Ultra-Lightweight Face Detection models to DetectionPrediction representation
    """
    __provider__ = 'ultra_lightweight_face_detection'

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
            'score_threshold': NumberField(value_type=float, min_value=0, max_value=1, default=0.7, optional=True,
                                           description='Minimal accepted score for valid boxes'),
        })

        return parameters

    def configure(self):
        self.scores_out = self.get_value_from_config('scores_out')
        self.boxes_out = self.get_value_from_config('boxes_out')
        self.score_threshold = self.get_value_from_config('score_threshold')
        self.outputs_verified = False

    def select_output_blob(self, outputs):
        self.scores_out = self.check_output_name(self.scores_out, outputs)
        self.boxes_out = self.check_output_name(self.boxes_out, outputs)
        self.outputs_verified = True

    def process(self, raw, identifiers, frame_meta):
        raw_outputs = self._extract_predictions(raw, frame_meta)
        if not self.outputs_verified:
            self.select_output_blob(raw_outputs)

        batch_scores = raw_outputs[self.scores_out]
        batch_boxes = raw_outputs[self.boxes_out]

        result = []
        for identifier, scores, boxes in zip(identifiers, batch_scores, batch_boxes):
            x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
            score = np.transpose(scores)[1]
            mask = score > self.score_threshold
            filtered_boxes, filtered_score = boxes[mask, :], score[mask]
            if filtered_score.size != 0:
                x_mins, y_mins, x_maxs, y_maxs = filtered_boxes.T
            labels = np.full_like(filtered_score, 1, dtype=int)

            result.append(DetectionPrediction(identifier, labels, filtered_score, x_mins, y_mins, x_maxs, y_maxs))

        return result


class PPDetectionAdapter(Adapter):
    __provider__ = 'ppdetection'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'boxes_out': StringField(description='output with boxes'),
            'num_boxes_out': StringField(description='number of boxes output')
        })
        return params

    def configure(self):
        self.boxes_out = self.get_value_from_config('boxes_out')
        self.num_boxes_out = self.get_value_from_config('num_boxes_out')
        self.outputs_verified = False

    def select_output_blob(self, outputs):
        self.boxes_out = self.check_output_name(self.boxes_out, outputs)
        self.num_boxes_out = self.check_output_name(self.num_boxes_out, outputs)
        self.outputs_verified = True

    def process(self, raw, identifiers, frame_meta):
        predictions = self._extract_predictions(raw, frame_meta)
        if not self.outputs_verified:
            self.select_output_blob(predictions)
        results = []
        boxes_start = 0
        for identifier, num_boxes in zip(identifiers, predictions[self.num_boxes_out]):
            batch_boxes = predictions[self.boxes_out][boxes_start:num_boxes]
            boxes_start += num_boxes
            labels, scores, x_mins, y_mins, x_maxs, y_maxs = batch_boxes.T
            results.append(DetectionPrediction(identifier, labels, scores, x_mins, y_mins, x_maxs, y_maxs))

        return results


class NanoDetAdapter(Adapter):
    __provider__ = 'nanodet'

    @classmethod
    def validate_config(cls, config, fetch_only=False, **kwargs):
        return super().validate_config(
            config, fetch_only=fetch_only, on_extra_argument=ConfigValidator.ERROR_ON_EXTRA_ARGUMENT
        )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'num_classes': NumberField(
                description="Number of classes.", value_type=int, min_value=0, default=80, optional=True),
            'confidence_threshold': NumberField(optional=True, default=0.05, description="confidence threshold"),
            'nms_threshold': NumberField(optional=True, default=0.6, description="NMS threshold"),
            'max_detections': NumberField(optional=True, value_type=int, default=100,
                                          description="maximal number of detections"),
            'reg_max': NumberField(description="max value of integral set", value_type=int, min_value=0, default=7,
                                   optional=True),
            'strides': ListField(value_type=int, optional=True, default=[8, 16, 32],
                                 description='strides of input multi-level feature maps'),
            'is_legacy': BoolField(optional=True, default=False, description='using a legacy NanoDet model')
        })
        return parameters

    def configure(self):
        self.num_classes = self.get_value_from_config('num_classes')
        self.confidence_threshold = self.get_value_from_config('confidence_threshold')
        self.nms_threshold = self.get_value_from_config('nms_threshold')
        self.max_detections = self.get_value_from_config('max_detections')
        self.reg_max = self.get_value_from_config('reg_max')
        self.strides = self.get_value_from_config('strides')
        self.is_legacy = self.get_value_from_config('is_legacy')

    @staticmethod
    def distance2bbox(points, distance, max_shape):
        x1 = np.expand_dims(points[:, 0] - distance[:, 0], -1).clip(0, max_shape[1])
        y1 = np.expand_dims(points[:, 1] - distance[:, 1], -1).clip(0, max_shape[0])
        x2 = np.expand_dims(points[:, 0] + distance[:, 2], -1).clip(0, max_shape[1])
        y2 = np.expand_dims(points[:, 1] + distance[:, 3], -1).clip(0, max_shape[0])
        return np.concatenate((x1, y1, x2, y2), axis=-1)

    def get_single_level_center_point(self, featmap_size, stride):
        h, w = featmap_size
        ad = 0.5 if self.is_legacy else 0
        x_range, y_range = (np.arange(w) + ad) * stride, (np.arange(h) + ad) * stride
        y, x = np.meshgrid(y_range, x_range, indexing='ij')
        return y.flatten(), x.flatten()

    def get_bboxes(self, reg_preds, input_height, input_width):
        featmap_sizes = [(math.ceil(input_height / stride), math.ceil(input_width) / stride) for stride in self.strides]
        list_center_priors = []
        for stride, featmap_size in zip(self.strides, featmap_sizes):
            y, x = self.get_single_level_center_point(featmap_size, stride)
            strides = np.full_like(x, stride)
            list_center_priors.append(np.stack([x, y, strides, strides], axis=-1))
        center_priors = np.concatenate(list_center_priors, axis=0)
        dist_project = np.linspace(0, self.reg_max, self.reg_max + 1)
        x = np.dot(softmax(np.reshape(reg_preds, (*reg_preds.shape[:-1], 4, self.reg_max + 1)), -1), dist_project)
        dis_preds = x * np.expand_dims(center_priors[:, 2], -1)
        return self.distance2bbox(center_priors[:, :2], dis_preds, (input_height, input_width))

    def process(self, raw, identifiers, frame_meta):
        raw_output = self._extract_predictions(raw, frame_meta)
        self.select_output_blob(raw_output)
        raw_output = raw_output[self.output_blob]

        result = []
        for identifier, output, meta in zip(identifiers, raw_output, frame_meta):
            cls_scores = output[:, :self.num_classes]
            bbox_preds = output[:, self.num_classes:]
            input_height, input_width = meta['preferable_height'], meta['preferable_width']

            bboxes = self.get_bboxes(bbox_preds, input_height, input_width)
            detections = {'labels': [], 'scores': [], 'x_mins': [], 'y_mins': [], 'x_maxs': [], 'y_maxs': []}
            for label, score in enumerate(np.transpose(cls_scores)):
                mask = score > self.confidence_threshold
                filtered_boxes, score = bboxes[mask, :], score[mask]
                if score.size == 0:
                    continue
                x_mins, y_mins, x_maxs, y_maxs = filtered_boxes.T
                keep = NMS.nms(x_mins, y_mins, x_maxs, y_maxs, score, self.nms_threshold, include_boundaries=True)
                score = score[keep]
                x_mins, y_mins, x_maxs, y_maxs = x_mins[keep], y_mins[keep], x_maxs[keep], y_maxs[keep]
                labels = np.full_like(score, label, dtype=int)
                detections['labels'].extend(labels)
                detections['scores'].extend(score)
                detections['x_mins'].extend(x_mins)
                detections['y_mins'].extend(y_mins)
                detections['x_maxs'].extend(x_maxs)
                detections['y_maxs'].extend(y_maxs)
            if len(detections['scores']) > self.max_detections:
                sort_idx = np.argsort(detections['scores'])[::-1][:self.max_detections]
                for key, value in detections.items():
                    detections[key] = np.array(value)[sort_idx]
            result.append(
                DetectionPrediction(identifier, detections['labels'], detections['scores'], detections['x_mins'],
                                    detections['y_mins'], detections['x_maxs'], detections['y_maxs']))
        return result
