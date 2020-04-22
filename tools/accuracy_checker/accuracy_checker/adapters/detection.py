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

import itertools
import math
import re
import warnings
import numpy as np

from ..topology_types import SSD, FasterRCNN
from ..adapters import Adapter
from ..config import ConfigValidator, NumberField, StringField, ConfigError, ListField, BoolField
from ..postprocessor.nms import NMS
from ..representation import DetectionPrediction, ContainerPrediction


class SSDAdapter(Adapter):
    """
    Class for converting output of SSD model to DetectionPrediction representation
    """
    __provider__ = 'ssd'
    prediction_types = (DetectionPrediction, )
    topology_types = (SSD, FasterRCNN, )

    def process(self, raw, identifiers=None, frame_meta=None):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
        Returns:
            list of DetectionPrediction objects
        """
        prediction_batch = self._extract_predictions(raw, frame_meta)[self.output_blob]
        prediction_count = prediction_batch.shape[2] if len(prediction_batch.shape) > 2 else prediction_batch.shape[0]
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


class PyTorchSSDDecoder(Adapter):
    """
    Class for converting output of PyTorch SSD models to DetectionPrediction representation
    """
    __provider__ = 'pytorch_ssd_decoder'

    def validate_config(self):
        super().validate_config(on_extra_argument=ConfigValidator.ERROR_ON_EXTRA_ARGUMENT)

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
        self.confidence_threshold = self.get_value_from_config('confidence_threshold')
        self.nms_threshold = self.get_value_from_config('nms_threshold')
        self.keep_top_k = self.get_value_from_config('keep_top_k')
        self.do_softmax = self.get_value_from_config('do_softmax')
        feat_size = self.get_value_from_config('feat_size')

        # Set default values according to:
        # https://github.com/mlperf/inference/tree/master/cloud/single_stage_detector
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.feat_size = [[50, 50], [25, 25], [13, 13], [7, 7], [3, 3], [3, 3]] if feat_size is None else feat_size
        self.scales = [21, 45, 99, 153, 207, 261, 315]
        self.strides = [3, 3, 2, 2, 2, 2]
        self.scale_xy = 0.1
        self.scale_wh = 0.2

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

    def process(self, raw, identifiers=None, frame_meta=None):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
        Returns:
            list of DetectionPrediction objects
        """

        raw_outputs = self._extract_predictions(raw, frame_meta)

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


class TFObjectDetectionAPIAdapter(Adapter):
    """
    Class for converting output of SSD model to DetectionPrediction representation
    """
    __provider__ = 'tf_object_detection'

    def validate_config(self):
        super().validate_config(on_extra_argument=ConfigValidator.ERROR_ON_EXTRA_ARGUMENT)

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
        self.classe_out = self.get_value_from_config('classes_out')
        self.boxes_out = self.get_value_from_config('boxes_out')
        self.scores_out = self.get_value_from_config('scores_out')
        self.num_detections_out = self.get_value_from_config('num_detections_out')

    def process(self, raw, identifiers=None, frame_meta=None):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
        Returns:
            list of DetectionPrediction objects
        """
        prediction_batch = self._extract_predictions(raw, frame_meta)
        classes_batch = prediction_batch[self.classe_out]
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

    def validate_config(self):
        super().validate_config(on_extra_argument=ConfigValidator.ERROR_ON_EXTRA_ARGUMENT)

    def configure(self):
        self.face_detection_out = self.launcher_config['face_out']
        self.person_detection_out = self.launcher_config['person_out']
        self.face_adapter = SSDAdapter(self.launcher_config, self.label_map, self.face_detection_out)
        self.person_adapter = SSDAdapter(self.launcher_config, self.label_map, self.person_detection_out)

    def process(self, raw, identifiers=None, frame_meta=None):
        face_batch_result = self.face_adapter.process(raw, identifiers)
        person_batch_result = self.person_adapter.process(raw, identifiers)
        result = [ContainerPrediction({self.face_detection_out: face_result, self.person_detection_out: person_result})
                  for face_result, person_result in zip(face_batch_result, person_batch_result)]

        return result


class SSDAdapterMxNet(Adapter):
    """
    Class for converting output of MXNet SSD model to DetectionPrediction representation
    """
    __provider__ = 'ssd_mxnet'

    def process(self, raw, identifiers=None, frame_meta=None):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model which is ndarray of shape (batch, det_count, 6),
                 each detection is defined by 6 values: class_id, prob, x_min, y_min, x_max, y_max
        Returns:
            list of DetectionPrediction objects
        """
        raw_outputs = self._extract_predictions(raw, frame_meta)
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
                'scores_out': StringField(description='name (or regex for it) of output layer with scores'),
                'bboxes_out': StringField(description='name (or regex for it) of output layer with bboxes')
            }
        )
        return parameters

    def configure(self):
        self.labels_out = self.get_value_from_config('labels_out')
        self.scores_out = self.get_value_from_config('scores_out')
        self.bboxes_out = self.get_value_from_config('bboxes_out')
        self.outputs_verified = False

    def process(self, raw, identifiers=None, frame_meta=None):
        raw_outputs = self._extract_predictions(raw, frame_meta)
        results = []
        if not self.outputs_verified:
            self._get_output_names(raw_outputs)
        for identifier, bboxes, scores, labels in zip(
                identifiers, raw_outputs[self.bboxes_out], raw_outputs[self.scores_out], raw_outputs[self.labels_out]
        ):
            x_mins, y_mins, x_maxs, y_maxs = bboxes.T
            results.append(DetectionPrediction(identifier, labels, scores, x_mins, y_mins, x_maxs, y_maxs))

        return results

    def _get_output_names(self, raw_outputs):
        labels_regex = re.compile(self.labels_out)
        scores_regex = re.compile(self.scores_out)
        bboxes_regex = re.compile(self.bboxes_out)

        def find_layer(regex, output_name, all_outputs):
            suitable_layers = [layer_name for layer_name in all_outputs if regex.match(layer_name)]
            if not suitable_layers:
                raise ValueError('suitable layer for {} output is not found'.format(output_name))

            if len(suitable_layers) > 1:
                raise ValueError('more than 1 layers matched to regular expression, please specify more detailed regex')

            return suitable_layers[0]

        self.labels_out = find_layer(labels_regex, 'labels', raw_outputs)
        self.scores_out = find_layer(scores_regex, 'scores', raw_outputs)
        self.bboxes_out = find_layer(bboxes_regex, 'bboxes', raw_outputs)

        self.outputs_verified = True


class MTCNNPAdapter(Adapter):
    __provider__ = 'mtcnn_p'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update(
            {
                'probability_out': StringField(description='Name of Output layer with detection boxes probabilities'),
                'region_out': StringField(description='Name of output layer with detected regions'),
                'regions_format': StringField(
                    optional=True, choices=['hw', 'wh'], default='wh',
                    description='determination of coordinates order in regions, wh uses order x1y1x2y2, hw - y1x1y2x2'
                )
            }
        )

        return parameters

    def configure(self):
        self.probability_out = self.get_value_from_config('probability_out')
        self.region_out = self.get_value_from_config('region_out')
        self.regions_format = self.get_value_from_config('regions_format')

    @staticmethod
    def nms(boxes, threshold, overlap_type):
        """
        Args:
          boxes: [:,0:5]
          threshold: 0.5 like
          overlap_type: 'Min' or 'Union'
        Returns:
            indexes of passed boxes
        """
        if boxes.shape[0] == 0:
            return np.array([])
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]
        area = np.multiply(x2 - x1 + 1, y2 - y1 + 1)
        inds = np.array(scores.argsort())

        pick = []
        while np.size(inds) > 0:
            xx1 = np.maximum(x1[inds[-1]], x1[inds[0:-1]])
            yy1 = np.maximum(y1[inds[-1]], y1[inds[0:-1]])
            xx2 = np.minimum(x2[inds[-1]], x2[inds[0:-1]])
            yy2 = np.minimum(y2[inds[-1]], y2[inds[0:-1]])
            width = np.maximum(0.0, xx2 - xx1 + 1)
            height = np.maximum(0.0, yy2 - yy1 + 1)
            inter = width * height
            if overlap_type == 'Min':
                overlap = inter / np.minimum(area[inds[-1]], area[inds[0:-1]])
            else:
                overlap = inter / (area[inds[-1]] + area[inds[0:-1]] - inter)
            pick.append(inds[-1])
            inds = inds[np.where(overlap <= threshold)[0]]

        return pick

    def process(self, raw, identifiers=None, frame_meta=None):
        total_boxes_batch = self._extract_predictions(raw, frame_meta)
        results = []
        for total_boxes, identifier in zip(total_boxes_batch, identifiers):
            if np.size(total_boxes) == 0:
                results.append(DetectionPrediction(identifier, [], [], [], [], [], []))
                continue
            pick = self.nms(total_boxes, 0.7, 'Union')
            total_boxes = total_boxes[pick]
            regh = total_boxes[:, 3] - total_boxes[:, 1]
            regw = total_boxes[:, 2] - total_boxes[:, 0]
            x_mins = total_boxes[:, 0] + total_boxes[:, 5] * regw
            y_mins = total_boxes[:, 1] + total_boxes[:, 6] * regh
            x_maxs = total_boxes[:, 2] + total_boxes[:, 7] * regw
            y_maxs = total_boxes[:, 3] + total_boxes[:, 8] * regh
            scores = total_boxes[:, 4]
            results.append(
                DetectionPrediction(identifier, np.full_like(scores, 1), scores, x_mins, y_mins, x_maxs, y_maxs)
            )


        return results

    @staticmethod
    def generate_bounding_box(mapping, reg, scale, t, r_format):
        stride = 2
        cellsize = 12
        mapping = mapping.T
        indexes = [0, 1, 2, 3] if r_format == 'wh' else [1, 0, 3, 2]
        dx1 = reg[indexes[0], :, :].T
        dy1 = reg[indexes[1], :, :].T
        dx2 = reg[indexes[2], :, :].T
        dy2 = reg[indexes[3], :, :].T
        (x, y) = np.where(mapping >= t)

        yy = y
        xx = x

        score = mapping[x, y]
        reg = np.array([dx1[x, y], dy1[x, y], dx2[x, y], dy2[x, y]])

        if reg.shape[0] == 0:
            pass
        bounding_box = np.array([yy, xx]).T

        bb1 = np.fix((stride * bounding_box + 1) / scale).T  # matlab index from 1, so with "boundingbox-1"
        bb2 = np.fix((stride * bounding_box + cellsize - 1 + 1) / scale).T  # while python don't have to
        score = np.array([score])

        bounding_box_out = np.concatenate((bb1, bb2, score, reg), axis=0)

        return bounding_box_out.T

    def _extract_predictions(self, outputs_list, meta):
        if not meta[0] or 'scales' not in meta[0]:
            return outputs_list[0]
        scales = meta[0]['scales']
        total_boxes = np.zeros((0, 9), np.float)
        for idx, outputs in enumerate(outputs_list):
            scale = scales[idx]
            mapping = outputs[self.probability_out][0, 1, :, :]
            regions = outputs[self.region_out][0]
            boxes = self.generate_bounding_box(mapping, regions, scale, 0.6, self.regions_format)
            if boxes.shape[0] != 0:
                pick = self.nms(boxes, 0.5, 'Union')

                if np.size(pick) > 0:
                    boxes = np.array(boxes)[pick, :]

            if boxes.shape[0] != 0:
                total_boxes = np.concatenate((total_boxes, boxes), axis=0)

        return [total_boxes]


class RetinaNetAdapter(Adapter):
    __provider__ = 'retinanet'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'loc_out': StringField(description='boxes localization output'),
            'class_out':  StringField(description="output with classes probabilities")
        })
        return params

    def configure(self):
        self.loc_out = self.get_value_from_config('loc_out')
        self.cls_out = self.get_value_from_config('class_out')
        self.pyramid_levels = [3, 4, 5, 6, 7]
        self.strides = [2 ** x for x in self.pyramid_levels]
        self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        self.ratios = np.array([0.5, 1, 2])
        self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
        self.std = np.array([0.1, 0.1, 0.2, 0.2])

    def process(self, raw, identifiers=None, frame_meta=None):
        raw_outputs = self._extract_predictions(raw, frame_meta)
        results = []
        for identifier, loc_pred, cls_pred, meta in zip(
                identifiers, raw_outputs[self.loc_out], raw_outputs[self.cls_out], frame_meta
        ):
            _, _, h, w = next(iter(meta.get('input_shape', {'data': (1, 3, 800, 800)}).values()))
            anchors = self.create_anchors([w, h])
            transformed_anchors = self.regress_boxes(anchors, loc_pred)
            labels, scores = np.argmax(cls_pred, axis=1), np.max(cls_pred, axis=1)
            scores_mask = np.reshape(scores > 0.05, -1)
            transformed_anchors = transformed_anchors[scores_mask, :]
            x_mins, y_mins, x_maxs, y_maxs = transformed_anchors.T
            results.append(DetectionPrediction(
                identifier, labels[scores_mask], scores[scores_mask], x_mins / w, y_mins / h, x_maxs / w, y_maxs / h
            ))

        return results

    def create_anchors(self, input_shape):
        def _generate_anchors(base_size=16):
            """
            Generate anchor (reference) windows by enumerating aspect ratios X
            scales w.r.t. a reference window.
            """
            num_anchors = len(self.ratios) * len(self.scales)
            # initialize output anchors
            anchors = np.zeros((num_anchors, 4))
            # scale base_size
            anchors[:, 2:] = base_size * np.tile(self.scales, (2, len(self.ratios))).T
            # compute areas of anchors
            areas = anchors[:, 2] * anchors[:, 3]
            # correct for ratios
            anchors[:, 2] = np.sqrt(areas / np.repeat(self.ratios, len(self.scales)))
            anchors[:, 3] = anchors[:, 2] * np.repeat(self.ratios, len(self.scales))
            # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
            anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
            anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

            return anchors

        def _shift(shape, stride, anchors):
            shift_x = (np.arange(0, shape[1]) + 0.5) * stride
            shift_y = (np.arange(0, shape[0]) + 0.5) * stride
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)

            shifts = np.vstack((
                shift_x.ravel(), shift_y.ravel(),
                shift_x.ravel(), shift_y.ravel()
            )).transpose()
            a = anchors.shape[0]
            k = shifts.shape[0]
            all_anchors = (anchors.reshape((1, a, 4)) + shifts.reshape((1, k, 4)).transpose((1, 0, 2)))
            all_anchors = all_anchors.reshape((k * a, 4))

            return all_anchors

        image_shapes = [(np.array(input_shape) + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]
        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4)).astype(np.float32)
        for idx, _ in enumerate(self.pyramid_levels):
            anchors = _generate_anchors(base_size=self.sizes[idx])
            shifted_anchors = _shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

        return all_anchors

    def regress_boxes(self, boxes, deltas):
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        dx = deltas[:, 0] * self.std[0]
        dy = deltas[:, 1] * self.std[1]
        dw = deltas[:, 2] * self.std[2]
        dh = deltas[:, 3] * self.std[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = np.exp(dw) * widths
        pred_h = np.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = np.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], axis=1)

        return pred_boxes


class ClassAgnosticDetectionAdapter(Adapter):
    """
    Class for converting 'boxes' [n,5] output of detection model to
    DetectionPrediction representation
    """
    __provider__ = 'class_agnostic_detection'
    prediction_types = (DetectionPrediction, )

    def validate_config(self):
        super().validate_config(on_extra_argument=ConfigValidator.ERROR_ON_EXTRA_ARGUMENT)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'output_blob': StringField(optional=True, default=None, description="Output blob name."),
            'scale': NumberField(optional=True, default=1.0, description="Scale factor for bboxes."),
        })

        return parameters

    def configure(self):
        self.out_blob_name = self.get_value_from_config('output_blob')
        self.scale = self.get_value_from_config('scale')

    def process(self, raw, identifiers=None, frame_meta=None):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
        Returns:
            list of DetectionPrediction objects
        """
        predictions = self._extract_predictions(raw, frame_meta)
        if self.out_blob_name is None:
            self.out_blob_name = self._find_output(predictions)
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
        params.update(
            {
                'cls_out': StringField(description='bboxes predicted classes score out'),
                'bbox_out': StringField(
                    description='bboxes output with shape [N, 8]'
                ),
                'rois_out': StringField(description='rois features output')
            }
        )
        return params

    def configure(self):
        self.cls_out = self.get_value_from_config('cls_out')
        self.bbox_out = self.get_value_from_config('bbox_out')
        self.rois_out = self.get_value_from_config('rois_out')

    def process(self, raw, identifiers=None, frame_meta=None):
        raw_out = self._extract_predictions(raw, frame_meta)
        predicted_classes = raw_out[self.cls_out]
        predicted_deltas = raw_out[self.bbox_out]
        predicted_proposals = raw_out[self.rois_out]
        x_scale = frame_meta[0]['scale_x']
        y_scale = frame_meta[0]['scale_y']
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
        num_classes = predicted_classes.shape[-1] - 1 # skip background
        x_mins, y_mins, x_maxs, y_maxs = predicted_boxes[:, 4:].T
        detections = {'labels': [], 'scores': [], 'x_mins': [], 'y_mins': [], 'x_maxs': [], 'y_maxs': []}
        for cls_id in range(num_classes):
            cls_scores = predicted_classes[:, cls_id+1]
            keep = NMS.nms(x_mins, y_mins, x_maxs, y_maxs, cls_scores, 0.3, include_boundaries=False)
            filtered_score = cls_scores[keep]
            x_cls_mins = x_mins[keep]
            y_cls_mins = y_mins[keep]
            x_cls_maxs = x_maxs[keep]
            y_cls_maxs = y_maxs[keep]
            # Save detections
            labels = np.full_like(filtered_score, cls_id+1)
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
