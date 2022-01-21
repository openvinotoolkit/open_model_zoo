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

        result = []
        for identifier in identifiers:
            prediction_mask = np.where(prediction_batch[:, -1] > 0.0)
            valid_detections = prediction_batch[prediction_mask]

            bboxes = self.scale * valid_detections[:, :-1]
            scores = valid_detections[:, -1]
            labels = np.ones([len(scores)], dtype=np.int32)

            result.append(DetectionPrediction(identifier, labels, scores, *zip(*bboxes)))

        return result

    @staticmethod
    def _find_output(predictions):
        filter_outputs = [
            output_name for output_name, out_data in predictions.items()
            if len(np.shape(out_data)) == 2 and np.shape(out_data)[-1] == 5
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

class HandLandmarkAdapter(Adapter):
    __provider__ = 'hand_landmark'

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

        def softmax(logits):
            res = [np.exp(logit) / np.sum(np.exp(logit)) for logit in logits]
            return np.array(res)

        for identifier, logits, boxes in zip(identifiers, raw_output[self.scores_out], raw_output[self.boxes_out]):
            x_mins, y_mins, x_maxs, y_maxs = box_cxcywh_to_xyxy(boxes)
            scores = softmax(logits)
            labels = np.argmax(scores[:, :-1], axis=-1)
            det_scores = np.max(scores[:, :-1], axis=-1)
            result.append(DetectionPrediction(identifier, labels, det_scores, x_mins, y_mins, x_maxs, y_maxs))

        return result

class Anchor:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0

    def set_x_center(self, x_center):
        self.x = x_center

    def set_y_center(self, y_center):
        self.y = y_center

    def set_w(self, width):
        self.width = width

    def set_h(self, height):
        self.height = height


class PalmDetectionAdapter(Adapter):
    __provider__ = 'palm_detection'

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

        # num_layers: 4
        # min_scale: 0.1484375
        # max_scale: 0.75
        # input_size_width: 192
        # input_size_height: 192
        # anchor_offset_x: 0.5
        # anchor_offset_y: 0.5
        # strides: 8
        # strides: 16
        # strides: 16
        # strides: 16
        # aspect_ratios: 1.0
        # fixed_anchor_size: true

        # options: {
        #     [mediapipe.SsdAnchorsCalculatorOptions.ext]
        # {
        #     num_layers: 4
        #     min_scale: 0.1484375
        #     max_scale: 0.75
        #     input_size_height: 128
        #     input_size_width: 128
        #     anchor_offset_x: 0.5
        #     anchor_offset_y: 0.5
        #     strides: 8
        #     strides: 16
        #     strides: 16
        #     strides: 16
        #     aspect_ratios: 1.0
        #     fixed_anchor_size: true
        # }

        self.num_layers = 4
        self.min_scale = 0.1484375
        self.max_scale = 0.75
        self.input_size_height = 128
        self.input_size_width = 128
        self.anchor_offset_x = 0.5
        self.anchor_offset_y = 0.5
        self.strides = [8, 16, 16, 16]
        self.aspect_ratios = [1.0,]
        self.reduce_boxes_in_lowest_layer = False
        self.feature_map_height = []
        self.feature_map_width = []
        self.inteprolated_scale_aspect_ratio = 1.0
        self.fixed_anchor_size = True

        self.anchors = self.generate_anchors()

        self.sigmoid_score = True
        self.score_clipping_thresh = 100.0
        self.has_score_clipping_thresh = self.score_clipping_thresh != 0
        self.reverse_output_order = True
        self.keypoint_coord_offset = 4
        self.num_keypoints = 7
        self.num_values_per_keypoint = 2

        self.x_scale = 128.0
        self.y_scale = 128.0
        self.w_scale = 128.0
        self.h_scale = 128.0
        self.min_score_thresh = 0.5
        self.has_min_score_thresh = self.min_score_thresh != 0
        self.apply_exponential_on_box_size = False
        self.num_classes = 1

        self.min_suppression_threshold = 0.3

    def select_output_blob(self, outputs):
        self.scores_out = self.check_output_name(self.scores_out, outputs)
        self.boxes_out = self.check_output_name(self.boxes_out, outputs)
        self.outputs_verified = True

    def process(self, raw, identifiers, frame_meta):
        result = []
        raw_output = self._extract_predictions(raw, frame_meta)
        if not self.outputs_verified:
            self.select_output_blob(raw_output)

        # def box_cxcywh_to_xyxy(x):
        #     x_c, y_c, w, h = x.T
        #     b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
        #          (x_c + 0.5 * w), (y_c + 0.5 * h)]
        #     return b
        #
        # def softmax(logits):
        #     res = [np.exp(logit) / np.sum(np.exp(logit)) for logit in logits]
        #     return np.array(res)

        for identifier, raw_scores, raw_boxes in zip(identifiers, raw_output[self.scores_out],
                                                     raw_output[self.boxes_out]):
            num_boxes, _ = raw_boxes.shape
            #     MP_RETURN_IF_ERROR(DecodeBoxes(raw_boxes, anchors_, &boxes));
            boxes = self.decode_boxes(raw_boxes)
            detection_scores = np.zeros(num_boxes)
            detection_classes = np.zeros(num_boxes)
            #
            #     std::vector<float> detection_scores(num_boxes_);
            #     std::vector<int> detection_classes(num_boxes_);
            #
            #     // Filter classes by scores.
            #     for (int i = 0; i < num_boxes_; ++i) {
            for i in range(num_boxes):
            #       int class_id = -1;
            #       float max_score = -std::numeric_limits<float>::max();
                class_id = -1
                max_score = -np.inf
            #       // Find the top score for box i.
            #       for (int score_idx = 0; score_idx < num_classes_; ++score_idx) {
                for score_idx in range(self.num_classes):
            #         if (ignore_classes_.find(score_idx) == ignore_classes_.end()) {
            #           auto score = raw_scores[i * num_classes_ + score_idx];
                    score = raw_scores[i, score_idx]
            #           if (options_.sigmoid_score()) {
                    if self.sigmoid_score:
            #             if (options_.has_score_clipping_thresh()) {
                        if self.has_score_clipping_thresh:
            #               score = score < -options_.score_clipping_thresh()
            #                           ? -options_.score_clipping_thresh()
            #                           : score;
                            score = -self.score_clipping_thresh if score < -self.score_clipping_thresh else score
            #               score = score > options_.score_clipping_thresh()
            #                           ? options_.score_clipping_thresh()
            #                           : score;
                            score = self.score_clipping_thresh if score > self.score_clipping_thresh else score
            #             }
            #             score = 1.0f / (1.0f + std::exp(-score));
                        score = 1.0 / (1.0 + np.exp(-score))
            #           }
            #           if (max_score < score) {
            #             max_score = score;
            #             class_id = score_idx;
            #           }
                    if max_score < score:
                        max_score = score
                        class_id = score_idx
            #         }
            #       }
            #       detection_scores[i] = max_score;
            #       detection_classes[i] = class_id;
                detection_classes[i] = class_id
                detection_scores[i] = max_score
            #     }
            #
            #     MP_RETURN_IF_ERROR(
            #         ConvertToDetections(boxes.data(), detection_scores.data(),
            #                             detection_classes.data(), output_detections));
            detections = self.convert_to_detections(boxes, detection_scores, detection_classes)
            # x_mins, y_mins, x_maxs, y_maxs = box_cxcywh_to_xyxy(boxes)
            # scores = softmax(logits)

            # NMS decoding

            detections = self.do_nms(detections)

            x_mins = []
            y_mins = []
            x_maxs = []
            y_maxs = []
            labels = []
            det_scores = []
            for detection in detections:
                bbox = detection.location_data.relative_bounding_box()
                x_mins.append(bbox.xmin * self.x_scale)
                y_mins.append(bbox.ymin * self.x_scale)
                x_maxs.append((bbox.xmin + bbox.width) * self.x_scale)
                y_maxs.append((bbox.ymin + bbox.height) * self.x_scale)
                labels.append(detection.label_id[0])
                det_scores.append(detection.score[0])

            result.append(DetectionPrediction(identifier, labels, det_scores, x_mins, y_mins, x_maxs, y_maxs))

        return result


# absl::Status    Process(CalculatorContext * cc) override
    def do_nms(self, detections):
#     {
#     // Add all input detections to the same vector.
#         Detections input_detections;
#     for (int i = 0; i < options_.num_detection_streams(); ++i) {
#         const auto & detections_packet = cc->Inputs().Index(i).Value();
#         // Check whether this stream has a packet for this timestamp.
#         if (detections_packet.IsEmpty()) {
#         continue;
#         }
#         const auto & detections = detections_packet.Get < Detections > ();
#         input_detections.insert(input_detections.end(), detections.begin(),
#                                 detections.end());
#     }
#
#     // Check if there are any detections at all.
#     if (input_detections.empty())
#     {
#         if (options_.return_empty_detections())
#         {
#
#
#             cc->Outputs().Index(0).Add(new
#             Detections(), cc->InputTimestamp());
#         }
#         return absl::OkStatus();
#     }
#
#     // Remove all but the maximum scoring label from each input detection.This
#     // corresponds to non - maximum suppression among detections which have
#     // identical locations.
#     Detections pruned_detections;
#     pruned_detections.reserve(input_detections.size());
#     for (auto & detection : input_detections)
#     {
#         if (RetainMaxScoringLabelOnly( & detection)) {
#             pruned_detections.push_back(detection);
#         }
#     }
        pruned_detections = [detection for detection in detections if detection.RetainMaxScoringLabelOnly()]
#
# // Copy all the scores (there is a single score in each detection after
# // the above pruning) to an indexed vector for sorting. The first value is
# // the index of the detection in the original vector from which the score
# // stems, while the second is the actual score.
#     IndexedScores indexed_scores;
#     indexed_scores.reserve(pruned_detections.size());
#     for (int index = 0; index < pruned_detections.size(); ++index) {
#         indexed_scores.push_back(std::make_pair(index, pruned_detections[index].score(0)));
#     }
#     std::sort(indexed_scores.begin(), indexed_scores.end(), SortBySecond);
        indexed_scores = [(i, pruned_detections[i].score[0]) for i in range(len(pruned_detections))]
        indexed_scores.sort(key=lambda x: x[1])
#
#     const int max_num_detections = (options_.max_num_detections() > -1) ? options_.max_num_detections() : static_cast < int > (indexed_scores.size());
# // A set of detections and locations, wrapping the location data from each
# // detection, which are retained after the non - maximum suppression.
#     auto * retained_detections = new Detections();
#     retained_detections->reserve(max_num_detections);
#
#     if (options_.algorithm() == NonMaxSuppressionCalculatorOptions::WEIGHTED) {
#         WeightedNonMaxSuppression(indexed_scores, pruned_detections,
#         max_num_detections, cc, retained_detections);
#     }
#     else {
#         NonMaxSuppression(indexed_scores, pruned_detections, max_num_detections,
#         cc, retained_detections);
#     }
#
        retained_detections = self.WeightedNMS(indexed_scores, pruned_detections)
#     cc->Outputs().Index(0).Add(retained_detections, cc->InputTimestamp());
#
#     return absl::OkStatus();
        return retained_detections
# }

  # void WeightedNonMaxSuppression(const IndexedScores& indexed_scores,
  #                                const Detections& detections,
  #                                int max_num_detections, CalculatorContext* cc,
  #                                Detections* output_detections) {
    def WeightedNMS(self, indexed_scores, detections):
  #   IndexedScores remained_indexed_scores;
  #   remained_indexed_scores.assign(indexed_scores.begin(),
  #                                  indexed_scores.end());
  #
  #   IndexedScores remained;
  #   IndexedScores candidates;
  #   output_detections->clear();
        remained_indexed_scores = [t for t in indexed_scores]
        output_detections = []
#   while (!remained_indexed_scores.empty()) {
        while len(remained_indexed_scores):
  #     const int original_indexed_scores_size = remained_indexed_scores.size();
  #     const auto& detection = detections[remained_indexed_scores[0].first];
  #     if (options_.min_score_threshold() > 0 &&
  #         detection.score(0) < options_.min_score_threshold()) {
  #       break;
  #     }
  #     remained.clear();
  #     candidates.clear();
  #     const Location location(detection.location_data());
            original_indexed_scores_size = len(remained_indexed_scores)
            detection = detections[remained_indexed_scores[0][0]]
            if self.min_score_thresh > 0 and detection.score[0] < self.min_score_thresh:
                break
            remained = []
            candidates = []
            location = detection.location_data
  #     // This includes the first box.
  #     for (const auto& indexed_score : remained_indexed_scores) {
            for indexed_score in remained_indexed_scores:
  #       Location rest_location(detections[indexed_score.first].location_data());
  #       float similarity =
  #           OverlapSimilarity(options_.overlap_type(), rest_location, location);
  #       if (similarity > options_.min_suppression_threshold()) {
  #         candidates.push_back(indexed_score);
  #       } else {
  #         remained.push_back(indexed_score);
  #       }
                rest_location = detections[indexed_score[0]].location_data
                similarity = self.overlap_similarity_iou(rest_location, location)
                if similarity > self.min_suppression_threshold:
                    candidates.append(indexed_score)
                else:
                    remained.append(indexed_score)
  #     }
  #     auto weighted_detection = detection;
            weighted_detection = Detection()
            weighted_detection.add_label_id(detection.label_id[0])
            weighted_detection.add_score(detection.score[0])
  #     if (!candidates.empty()) {
            if len(candidates):
  #       const int num_keypoints =
  #           detection.location_data().relative_keypoints_size();
  #       std::vector<float> keypoints(num_keypoints * 2);
  #       float w_xmin = 0.0f;
  #       float w_ymin = 0.0f;
  #       float w_xmax = 0.0f;
  #       float w_ymax = 0.0f;
  #       float total_score = 0.0f;
                w_xmin = 0
                w_xmax = 0
                w_ymin = 0
                w_ymax = 0
                total_score = 0
                keypoints = [[0, 0]] * self.num_keypoints
  #       for (const auto& candidate : candidates) {
                for candidate in candidates:
  #         total_score += candidate.second;
  #         const auto& location_data =
  #             detections[candidate.first].location_data();
  #         const auto& bbox = location_data.relative_bounding_box();
  #         w_xmin += bbox.xmin() * candidate.second;
  #         w_ymin += bbox.ymin() * candidate.second;
  #         w_xmax += (bbox.xmin() + bbox.width()) * candidate.second;
  #         w_ymax += (bbox.ymin() + bbox.height()) * candidate.second;
  #
                    total_score += candidate[1]
                    location_data = detections[candidate[0]].location_data
                    bbox = location_data.relative_bounding_box()
                    w_xmin += bbox.xmin * candidate[1]
                    w_ymin += bbox.ymin * candidate[1]
                    w_xmax += (bbox.xmin + bbox.width) * candidate[1]
                    w_ymax += (bbox.ymin + bbox.height) * candidate[1]

#         for (int i = 0; i < num_keypoints; ++i) {
                    for i in range(self.num_keypoints):
  #           keypoints[i * 2] +=
  #               location_data.relative_keypoints(i).x() * candidate.second;
  #           keypoints[i * 2 + 1] +=
  #               location_data.relative_keypoints(i).y() * candidate.second;
                        keypoints[i][0] += location_data.keypoints[i].x * candidate[1]
                        keypoints[i][1] += location_data.keypoints[i].y * candidate[1]
  #         }
  #       }
  #       auto* weighted_location = weighted_detection.mutable_location_data()
  #                                     ->mutable_relative_bounding_box();

  #       weighted_location->set_xmin(w_xmin / total_score);
  #       weighted_location->set_ymin(w_ymin / total_score);
  #       weighted_location->set_width((w_xmax / total_score) -
  #                                    weighted_location->xmin());
  #       weighted_location->set_height((w_ymax / total_score) -
  #                                     weighted_location->ymin());
                weighted_location = weighted_detection.get_location_data()
                bbox = weighted_location.relative_bounding_box()
                bbox.set_xmin(w_xmin / total_score)
                bbox.set_ymin(w_ymin / total_score)
                bbox.set_width((w_xmax - w_xmin) / total_score)
                bbox.set_height((w_ymax - w_ymin) / total_score)

#       for (int i = 0; i < num_keypoints; ++i) {
  #         auto* keypoint = weighted_detection.mutable_location_data()
  #                              ->mutable_relative_keypoints(i);
  #         keypoint->set_x(keypoints[i * 2] / total_score);
  #         keypoint->set_y(keypoints[i * 2 + 1] / total_score);
  #       }
                for kpdata in keypoints:
                    kp = weighted_location.add_relative_keypoint()
                    kp.set_x(kpdata[0] / total_score)
                    kp.set_y(kpdata[1] / total_score)


    #     }
  #
  #     output_detections->push_back(weighted_detection);
                output_detections.append(weighted_detection)
  #     // Breaks the loop if the size of indexed scores doesn't change after an
  #     // iteration.
  #     if (original_indexed_scores_size == remained.size()) {
  #       break;
  #     } else {
  #       remained_indexed_scores = std::move(remained);
  #     }
            if original_indexed_scores_size == len(remained):
                break
            else:
                remained_indexed_scores = [t for t in remained]

        return output_detections
    #   }
  # }


# // Computes an overlap similarity between two rectangles. Similarity measure is
# // defined by overlap_type parameter.
# float OverlapSimilarity(
#     const NonMaxSuppressionCalculatorOptions::OverlapType overlap_type,
#     const Rectangle_f& rect1, const Rectangle_f& rect2) {
    def overlap_similarity_iou(self, rect1, rect2):
#   if (!rect1.Intersects(rect2)) return 0.0f;
        if not rect1.intersects(rect2):
            return 0
#   const float intersection_area = Rectangle_f(rect1).Intersect(rect2).Area();
#   float normalization;
        intersection_area = rect1.intersect(rect2).area()
#   switch (overlap_type) {
#     case NonMaxSuppressionCalculatorOptions::JACCARD:
#       normalization = Rectangle_f(rect1).Union(rect2).Area();
#       break;
#     case NonMaxSuppressionCalculatorOptions::MODIFIED_JACCARD:
#       normalization = rect2.Area();
#       break;
#     case NonMaxSuppressionCalculatorOptions::INTERSECTION_OVER_UNION:
#       normalization = rect1.Area() + rect2.Area() - intersection_area;
#       break;
#     default:
#       LOG(FATAL) << "Unrecognized overlap type: " << overlap_type;
#   }
        normalization = rect1.area() + rect2.area() - rect1.intersect(rect2).area()
#   return normalization > 0.0f ? intersection_area / normalization : 0.0f;
        return intersection_area / normalization if normalization > 0 else 0.
# }



    @staticmethod
    def calculate_scale(min_scale, max_scale, stride_index, num_strides):
        if num_strides == 1:
            return (min_scale + max_scale) * 0.5
        else:
            return min_scale + (max_scale - min_scale) * 1.0 * stride_index / (num_strides - 1)

    def generate_anchors(self):
        anchors = []
        layer_id = 0
        while (layer_id < self.num_layers):
            anchor_height = []
            anchor_width = []
            aspect_ratios = []
            scales = []

            last_same_stride_layer = layer_id
            while last_same_stride_layer < len(self.strides) and (self.strides[last_same_stride_layer] ==
                                                                  self.strides[layer_id]):
                scale = self.calculate_scale(self.min_scale, self.max_scale, last_same_stride_layer, len(self.strides))
                if (last_same_stride_layer == 0) and self.reduce_boxes_in_lowest_layer:
                    aspect_ratios.append(1.0)
                    aspect_ratios.append(2.0)
                    aspect_ratios.append(0.5)

                    scales.append(0.1)
                    scales.append(scale)
                    scales.append(scale)
                else:
                    for aspect_ratio_id in range(0, len(self.aspect_ratios)):
                        aspect_ratios.append(self.aspect_ratios[aspect_ratio_id])
                        scales.append(scale)
                if self.inteprolated_scale_aspect_ratio > 0.0:
                    scale_next = 1.0 if last_same_stride_layer == len(self.strides) - 1 else self.calculate_scale(self.min_scale,
                                                                                                                  self.max_scale,
                                                                                                                  last_same_stride_layer + 1,
                                                                                                                  len(self.strides))
                    scales.append(np.sqrt(scale * scale_next))
                    aspect_ratios.append(self.inteprolated_scale_aspect_ratio)
                last_same_stride_layer += 1

            for i in range(0, len(aspect_ratios)):
                ratio_sqrts = np.sqrt(aspect_ratios[i])
                anchor_height.append(scales[i] / ratio_sqrts)
                anchor_width.append(scales[i] * ratio_sqrts)
            feature_map_height = 0
            feature_map_width = 0
            if len(self.feature_map_height):
                feature_map_height = self.feature_map_height[layer_id]
                feature_map_width = self.feature_map_width[layer_id]
            else:
                stride = self.strides[layer_id]
                feature_map_height = int(np.ceil(1.0 * self.input_size_height / stride))
                feature_map_width = int(np.ceil(1.0 * self.input_size_width / stride))

            for y in range(0, feature_map_height):
                for x in range(0, feature_map_width):
                    for anchor_id in range(0, len(anchor_height)):
                        x_center = (x + self.anchor_offset_x) / feature_map_width
                        y_center = (y + self.anchor_offset_y) / feature_map_height

                        anchor = Anchor()
                        anchor.set_x_center(x_center)
                        anchor.set_y_center(y_center)

                        if self.fixed_anchor_size:
                            anchor.set_w(1.0)
                            anchor.set_h(1.0)
                        else:
                            anchor.set_w(anchor_width[anchor_id])
                            anchor.set_h(anchor_height[anchor_id])
                        anchors.append(anchor)

            layer_id = last_same_stride_layer

        return anchors

# void ConvertRawValuesToAnchors(const float* raw_anchors, int num_boxes,
#                                std::vector<Anchor>* anchors) {
#   anchors->clear();
#   for (int i = 0; i < num_boxes; ++i) {
#     Anchor new_anchor;
#     new_anchor.set_y_center(raw_anchors[i * kNumCoordsPerBox + 0]);
#     new_anchor.set_x_center(raw_anchors[i * kNumCoordsPerBox + 1]);
#     new_anchor.set_h(raw_anchors[i * kNumCoordsPerBox + 2]);
#     new_anchor.set_w(raw_anchors[i * kNumCoordsPerBox + 3]);
#     anchors->push_back(new_anchor);
# }
#
# void ConvertAnchorsToRawValues(const std::vector<Anchor>& anchors,
#                                int num_boxes, float* raw_anchors) {
#   CHECK_EQ(anchors.size(), num_boxes);
#   int box = 0;
#   for (const auto& anchor : anchors) {
#     raw_anchors[box * kNumCoordsPerBox + 0] = anchor.y_center();
#     raw_anchors[box * kNumCoordsPerBox + 1] = anchor.x_center();
#     raw_anchors[box * kNumCoordsPerBox + 2] = anchor.h();
#     raw_anchors[box * kNumCoordsPerBox + 3] = anchor.w();
#     ++box;
#   }
# }

# absl::Status TensorsToDetectionsCalculator::ProcessCPU(
#     CalculatorContext* cc, std::vector<Detection>* output_detections) {
#   const auto& input_tensors = *kInTensors(cc);
#
#   if (input_tensors.size() == 2 ||-
#       input_tensors.size() == kNumInputTensorsWithAnchors) {
#     // Postprocessing on CPU for model without postprocessing op. E.g. output
#     // raw score tensor and box tensor. Anchor decoding will be handled below.
#     // TODO: Add flexible input tensor size handling.
#     auto raw_box_tensor = &input_tensors[0];
#     RET_CHECK_EQ(raw_box_tensor->shape().dims.size(), 3);
#     RET_CHECK_EQ(raw_box_tensor->shape().dims[0], 1);
#     RET_CHECK_EQ(raw_box_tensor->shape().dims[1], num_boxes_);
#     RET_CHECK_EQ(raw_box_tensor->shape().dims[2], num_coords_);
#     auto raw_score_tensor = &input_tensors[1];
#     RET_CHECK_EQ(raw_score_tensor->shape().dims.size(), 3);
#     RET_CHECK_EQ(raw_score_tensor->shape().dims[0], 1);
#     RET_CHECK_EQ(raw_score_tensor->shape().dims[1], num_boxes_);
#     RET_CHECK_EQ(raw_score_tensor->shape().dims[2], num_classes_);
#     auto raw_box_view = raw_box_tensor->GetCpuReadView();
#     auto raw_boxes = raw_box_view.buffer<float>();
#     auto raw_scores_view = raw_score_tensor->GetCpuReadView();
#     auto raw_scores = raw_scores_view.buffer<float>();
#
#     // TODO: Support other options to load anchors.
#     if (!anchors_init_) {
#       if (input_tensors.size() == kNumInputTensorsWithAnchors) {
#         auto anchor_tensor = &input_tensors[2];
#         RET_CHECK_EQ(anchor_tensor->shape().dims.size(), 2);
#         RET_CHECK_EQ(anchor_tensor->shape().dims[0], num_boxes_);
#         RET_CHECK_EQ(anchor_tensor->shape().dims[1], kNumCoordsPerBox);
#         auto anchor_view = anchor_tensor->GetCpuReadView();
#         auto raw_anchors = anchor_view.buffer<float>();
#         ConvertRawValuesToAnchors(raw_anchors, num_boxes_, &anchors_);
#       } else if (!kInAnchors(cc).IsEmpty()) {
#         anchors_ = *kInAnchors(cc);
#       } else {
#         return absl::UnavailableError("No anchor data available.");
#       }
#       anchors_init_ = true;
#     }
#     std::vector<float> boxes(num_boxes_ * num_coords_);
#     MP_RETURN_IF_ERROR(DecodeBoxes(raw_boxes, anchors_, &boxes));
#
#     std::vector<float> detection_scores(num_boxes_);
#     std::vector<int> detection_classes(num_boxes_);
#
#     // Filter classes by scores.
#     for (int i = 0; i < num_boxes_; ++i) {
#       int class_id = -1;
#       float max_score = -std::numeric_limits<float>::max();
#       // Find the top score for box i.
#       for (int score_idx = 0; score_idx < num_classes_; ++score_idx) {
#         if (ignore_classes_.find(score_idx) == ignore_classes_.end()) {
#           auto score = raw_scores[i * num_classes_ + score_idx];
#           if (options_.sigmoid_score()) {
#             if (options_.has_score_clipping_thresh()) {
#               score = score < -options_.score_clipping_thresh()
#                           ? -options_.score_clipping_thresh()
#                           : score;
#               score = score > options_.score_clipping_thresh()
#                           ? options_.score_clipping_thresh()
#                           : score;
#             }
#             score = 1.0f / (1.0f + std::exp(-score));
#           }
#           if (max_score < score) {
#             max_score = score;
#             class_id = score_idx;
#           }
#         }
#       }
#       detection_scores[i] = max_score;
#       detection_classes[i] = class_id;
#     }
#
#     MP_RETURN_IF_ERROR(
#         ConvertToDetections(boxes.data(), detection_scores.data(),
#                             detection_classes.data(), output_detections));
#   } else {
#     // Postprocessing on CPU with postprocessing op (e.g. anchor decoding and
#     // non-maximum suppression) within the model.
#     RET_CHECK_EQ(input_tensors.size(), 4);
#
#     auto num_boxes_tensor = &input_tensors[3];
#     RET_CHECK_EQ(num_boxes_tensor->shape().dims.size(), 1);
#     RET_CHECK_EQ(num_boxes_tensor->shape().dims[0], 1);
#
#     auto detection_boxes_tensor = &input_tensors[0];
#     RET_CHECK_EQ(detection_boxes_tensor->shape().dims.size(), 3);
#     RET_CHECK_EQ(detection_boxes_tensor->shape().dims[0], 1);
#     const int max_detections = detection_boxes_tensor->shape().dims[1];
#     RET_CHECK_EQ(detection_boxes_tensor->shape().dims[2], num_coords_);
#
#     auto detection_classes_tensor = &input_tensors[1];
#     RET_CHECK_EQ(detection_classes_tensor->shape().dims.size(), 2);
#     RET_CHECK_EQ(detection_classes_tensor->shape().dims[0], 1);
#     RET_CHECK_EQ(detection_classes_tensor->shape().dims[1], max_detections);
#
#     auto detection_scores_tensor = &input_tensors[2];
#     RET_CHECK_EQ(detection_scores_tensor->shape().dims.size(), 2);
#     RET_CHECK_EQ(detection_scores_tensor->shape().dims[0], 1);
#     RET_CHECK_EQ(detection_scores_tensor->shape().dims[1], max_detections);
#
#     auto num_boxes_view = num_boxes_tensor->GetCpuReadView();
#     auto num_boxes = num_boxes_view.buffer<float>();
#     num_boxes_ = num_boxes[0];
#
#     auto detection_boxes_view = detection_boxes_tensor->GetCpuReadView();
#     auto detection_boxes = detection_boxes_view.buffer<float>();
#
#     auto detection_scores_view = detection_scores_tensor->GetCpuReadView();
#     auto detection_scores = detection_scores_view.buffer<float>();
#
#     auto detection_classes_view = detection_classes_tensor->GetCpuReadView();
#     auto detection_classes_ptr = detection_classes_view.buffer<float>();
#     std::vector<int> detection_classes(num_boxes_);
#     for (int i = 0; i < num_boxes_; ++i) {
#       detection_classes[i] = static_cast<int>(detection_classes_ptr[i]);
#     }
#     MP_RETURN_IF_ERROR(ConvertToDetections(detection_boxes, detection_scores,
#                                            detection_classes.data(),
#                                            output_detections));
#   }
#   return absl::OkStatus();
# }

# absl::Status TensorsToDetectionsCalculator::DecodeBoxes(
#     const float* raw_boxes, const std::vector<Anchor>& anchors,
#     std::vector<float>* boxes) {
    def decode_boxes(self, raw_boxes):
        boxes = []
        num_boxes, _ = raw_boxes.shape

#   for (int i = 0; i < num_boxes_; ++i) {
        for i in range(num_boxes):
#     const int box_offset = i * num_coords_ + options_.box_coord_offset();
#
#     float y_center = raw_boxes[box_offset];
#     float x_center = raw_boxes[box_offset + 1];
#     float h = raw_boxes[box_offset + 2];
#     float w = raw_boxes[box_offset + 3];
            y_center = raw_boxes[i, 0]
            x_center = raw_boxes[i, 1]
            h = raw_boxes[i, 2]
            w = raw_boxes[i, 3]
#     if (options_.reverse_output_order()) {
            if self.reverse_output_order:
#       x_center = raw_boxes[box_offset];
#       y_center = raw_boxes[box_offset + 1];
#       w = raw_boxes[box_offset + 2];
#       h = raw_boxes[box_offset + 3];
                x_center = raw_boxes[i, 0]
                y_center = raw_boxes[i, 1]
                w = raw_boxes[i, 2]
                h = raw_boxes[i, 3]
#     }
#
#     x_center =
#         x_center / options_.x_scale() * anchors[i].w() + anchors[i].x_center();
#     y_center =
#         y_center / options_.y_scale() * anchors[i].h() + anchors[i].y_center();
#
            x_center = x_center / self.x_scale * self.anchors[i].width + self.anchors[i].x
            y_center = y_center / self.y_scale * self.anchors[i].height + self.anchors[i].y

#     if (options_.apply_exponential_on_box_size()) {
#       h = std::exp(h / options_.h_scale()) * anchors[i].h();
#       w = std::exp(w / options_.w_scale()) * anchors[i].w();
#     } else {
#       h = h / options_.h_scale() * anchors[i].h();
#       w = w / options_.w_scale() * anchors[i].w();
#     }

            if self.apply_exponential_on_box_size:
                h = np.exp(h / self.h_scale) * self.anchors[i].height
                w = np.exp(w / self.w_scale) * self.anchors[i].width
            else:
                h = h / self.h_scale * self.anchors[i].height
                w = w / self.w_scale * self.anchors[i].width

#
#     const float ymin = y_center - h / 2.f;
#     const float xmin = x_center - w / 2.f;
#     const float ymax = y_center + h / 2.f;
#     const float xmax = x_center + w / 2.f;
#
#     (*boxes)[i * num_coords_ + 0] = ymin;
#     (*boxes)[i * num_coords_ + 1] = xmin;
#     (*boxes)[i * num_coords_ + 2] = ymax;
#     (*boxes)[i * num_coords_ + 3] = xmax;

            decoded = [y_center - h / 2, x_center - w / 2, y_center + h / 2, x_center + w / 2]
#
#     if (options_.num_keypoints()) {
#       for (int k = 0; k < options_.num_keypoints(); ++k) {
            for k in range(self.num_keypoints):
#         const int offset = i * num_coords_ + options_.keypoint_coord_offset() +
#                            k * options_.num_values_per_keypoint();
                offset = self.keypoint_coord_offset + k * self.num_values_per_keypoint
#
#         float keypoint_y = raw_boxes[offset];
#         float keypoint_x = raw_boxes[offset + 1];
                keypoint_y = raw_boxes[i, offset]
                keypoint_x = raw_boxes[i, offset + 1]
#         if (options_.reverse_output_order()) {
#           keypoint_x = raw_boxes[offset];
#           keypoint_y = raw_boxes[offset + 1];
#         }
                if self.reverse_output_order:
                    keypoint_x = raw_boxes[i, offset]
                    keypoint_y = raw_boxes[i, offset + 1]
#
#         (*boxes)[offset] = keypoint_x / options_.x_scale() * anchors[i].w() +
#                            anchors[i].x_center();
#         (*boxes)[offset + 1] =
#             keypoint_y / options_.y_scale() * anchors[i].h() +
#             anchors[i].y_center();
#       }
                decoded.append(keypoint_x / self.x_scale * self.anchors[i].width + self.anchors[i].x)
                decoded.append(keypoint_y / self.y_scale * self.anchors[i].height + self.anchors[i].y)
#     }
#   }
#
#   return absl::OkStatus();
# }
#
            boxes.append(decoded)

        return boxes

# absl::Status TensorsToDetectionsCalculator::ConvertToDetections(
#     const float* detection_boxes, const float* detection_scores,
#     const int* detection_classes, std::vector<Detection>* output_detections) {
    def convert_to_detections(self, detection_boxes, detection_scores, detection_classes):
        detections = []
#   for (int i = 0; i < num_boxes_; ++i) {
        for detection_box, detection_score, detection_class in zip(detection_boxes, detection_scores,
                                                                   detection_classes):
#     if (options_.has_min_score_thresh() &&
#         detection_scores[i] < options_.min_score_thresh()) {
#       continue;
#     }
            if self.has_min_score_thresh and detection_score < self.min_score_thresh:
                continue
#     const int box_offset = i * num_coords_;
#     Detection detection = ConvertToDetection(
#         detection_boxes[box_offset + 0], detection_boxes[box_offset + 1],
#         detection_boxes[box_offset + 2], detection_boxes[box_offset + 3],
#         detection_scores[i], detection_classes[i], options_.flip_vertically());
            detection = self.convert_to_detection(detection_box[0], detection_box[1], detection_box[2],
                                                  detection_box[3], detection_score, detection_class)
#     const auto& bbox = detection.location_data().relative_bounding_box();
#     if (bbox.width() < 0 || bbox.height() < 0 || std::isnan(bbox.width()) ||
#         std::isnan(bbox.height())) {
#       // Decoded detection boxes could have negative values for width/height due
#       // to model prediction. Filter out those boxes since some downstream-
#       // calculators may assume non-negative values. (b/171391719)
#       continue;
#     }
            bbox = detection.location_data.relative_bounding_box()
            if bbox.width < 0 or bbox.height < 0 or np.isnan(bbox.width) or np.isnan(bbox.height):
                continue
#     // Add keypoints.
#     if (options_.num_keypoints() > 0) {
#       auto* location_data = detection.mutable_location_data();
#       for (int kp_id = 0; kp_id < options_.num_keypoints() *
#                                       options_.num_values_per_keypoint();
            for i in range(self.num_keypoints):
#            kp_id += options_.num_values_per_keypoint()) {
#         auto keypoint = location_data->add_relative_keypoints();
                kp = detection.location_data.add_relative_keypoint()
#         const int keypoint_index =
#             box_offset + options_.keypoint_coord_offset() + kp_id;
#         keypoint->set_x(detection_boxes[keypoint_index + 0]);
#         keypoint->set_y(options_.flip_vertically()
#                             ? 1.f - detection_boxes[keypoint_index + 1]
#                             : detection_boxes[keypoint_index + 1]);
                kp.set_x(detection_box[self.keypoint_coord_offset + i * self.num_values_per_keypoint])
                kp.set_y(detection_box[self.keypoint_coord_offset + i * self.num_values_per_keypoint + 1])

#       }
#     }

            detections.append(detection)
#     output_detections->emplace_back(detection);
#   }
#   return absl::OkStatus();
# }
        return detections

#
# Detection TensorsToDetectionsCalculator::ConvertToDetection(
#     float box_ymin, float box_xmin, float box_ymax, float box_xmax, float score,
#     int class_id, bool flip_vertically) {
    def convert_to_detection(self, ymin, xmin, ymax, xmax, score, class_id):
#   Detection detection;
#   detection.add_score(score);
#   detection.add_label_id(class_id);
        detection = Detection()
        detection.add_score(score)
        detection.add_label_id(class_id)
#
#   LocationData* location_data = detection.mutable_location_data();
#   location_data->set_format(LocationData::RELATIVE_BOUNDING_BOX);
#/
#   LocationData::RelativeBoundingBox* relative_bbox =
#       location_data->mutable_relative_bounding_box();
#
        location_data = detection.get_location_data()
        relative_bbox = location_data.relative_bounding_box()

#   relative_bbox->set_xmin(box_xmin);
#   relative_bbox->set_ymin(flip_vertically ? 1.f - box_ymax : box_ymin);
#   relative_bbox->set_width(box_xmax - box_xmin);
#   relative_bbox->set_height(box_ymax - box_ymin);

        relative_bbox.set_xmin(xmin)
        relative_bbox.set_ymin(ymin)
        relative_bbox.set_width(xmax - xmin)
        relative_bbox.set_height(ymax - ymin)

#   return detection;
# }
        return detection


class Detection:
    def __init__(self):
        self.score = []
        self.label_id = []
        self.location_data = None

    def add_score(self, score):
        self.score.append(score)

    def add_label_id(self, label_id):
        self.label_id.append(label_id)

    def get_location_data(self):
        if self.location_data is None:
            self.location_data = Location()
        return self.location_data

    def clear_score(self):
        self.score = []


    def clear_label_id(self):
        self.label_id = []

    # // Removes all but the max scoring label and its score from the detection.
# // Returns true if the detection has at least one label.
# bool RetainMaxScoringLabelOnly(Detection* detection) {
    def RetainMaxScoringLabelOnly(self):
#   if (detection->label_id_size() == 0 && detection->label_size() == 0) {
#     return false;
#   }
        if len(self.label_id) == 0:
            return False
#   CHECK(detection->label_id_size() == detection->score_size() ||
#         detection->label_size() == detection->score_size())
#       << "Number of scores must be equal to number of detections.";
#
        assert len(self.label_id) == len(self.score)
#   std::vector<std::pair<int, float>> indexed_scores;
#   indexed_scores.reserve(detection->score_size());
#   for (int k = 0; k < detection->score_size(); ++k) {
#     indexed_scores.push_back(std::make_pair(k, detection->score(k)));
#   }
        indexed_scores = [(i, self.score[i]) for i in range(len(self.score))]

#   std::sort(indexed_scores.begin(), indexed_scores.end(), SortBySecond);
        indexed_scores.sort(key=lambda x: x[1])
#   const int top_index = indexed_scores[0].first;
#   detection->clear_score();
#   detection->add_score(indexed_scores[0].second);
        top_index = indexed_scores[0][0]
        self.clear_score()
        self.add_score(indexed_scores[0][1])
#   if (detection->label_id_size() > top_index) {
#     const int top_label_id = detection->label_id(top_index);
#     detection->clear_label_id();
#     detection->add_label_id(top_label_id);
#   } else {
#     const std::string top_label = detection->label(top_index);
#     detection->clear_label();
#     detection->add_label(top_label);
#   }
        top_label_id = self.label_id[top_index]
        self.clear_label_id()
        self.add_label_id(top_label_id)
#
#   return true;
        return True
# }


class Location:
    def __init__(self):
        self.bbox = None
        self.keypoints = []

    def relative_bounding_box(self):
        if self.bbox is None:
            self.bbox = RelativeBoundingBox()
        return self.bbox

    def add_relative_keypoint(self):
        kp = RelativeKeypoint()
        self.keypoints.append(kp)
        return kp

    def intersects(self, other):
        return ((self.bbox.xmin <= other.bbox.xmin and (self.bbox.xmin + self.bbox.width) >= other.bbox.xmin) or
                (other.bbox.xmin <= self.bbox.xmin and (other.bbox.xmin + other.bbox.width) >= self.bbox.xmin)) and (
            (self.bbox.ymin <= other.bbox.ymin and (self.bbox.ymin + self.bbox.height) >= other.bbox.ymin) or
            (other.bbox.ymin <= self.bbox.ymin and (other.bbox.ymin + other.bbox.width >= self.bbox.ymin)))

    def intersect(self, other):
        intersection = Location()
        ibox = intersection.relative_bounding_box()
        if self.intersects(other):
            bbox = self.bbox
            obox = other.bbox
            coords = [bbox.xmin, bbox.xmin + bbox.width, obox.xmin, obox.xmin + obox.width]
            coords.sort()
            ibox.set_xmin(coords[1])
            ibox.set_width(coords[2] - coords[1])
            coords = [bbox.ymin, bbox.ymin + bbox.height, obox.ymin, obox.ymin + obox.height]
            coords.sort()
            ibox.set_ymin(coords[1])
            ibox.set_height(coords[2] - coords[1])
        else:
            ibox.set_height(0.)
            ibox.set_width(0.)
            ibox.set_xmin(0.)
            ibox.set_ymin(0.)
        return intersection

    def area(self):
        return self.bbox.height * self.bbox.width

class RelativeBoundingBox:
    def __init__(self):
        self.xmin = 0
        self.ymin = 0
        self.width = 0
        self.height = 0

    def set_xmin(self, xmin):
        self.xmin = xmin

    def set_ymin(self, ymin):
        self.ymin = ymin

    def set_width(self, width):
        self.width = width

    def set_height(self, height):
        self.height = height


class RelativeKeypoint:
    def __init__(self):
        self.x = None
        self.y = None

    def set_x(self, x):
        self.x = x

    def set_y(self, y):
        self.y = y
