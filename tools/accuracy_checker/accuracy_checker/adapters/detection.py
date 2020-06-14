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
import warnings
import numpy as np

from ..adapters import Adapter
from ..config import ConfigValidator, NumberField, StringField, ConfigError
from ..postprocessor.nms import NMS
from ..representation import DetectionPrediction


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
        scales = [1] if not meta[0] or 'scales' not in meta[0] else meta[0]['scales']
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


class FaceBoxesAdapter(Adapter):
    """
    Class for converting output of FaceBoxes models to DetectionPrediction representation
    """
    __provider__ = 'faceboxes'

    def validate_config(self):
        super().validate_config(on_extra_argument=ConfigValidator.ERROR_ON_EXTRA_ARGUMENT)

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

        # Set default values
        self.min_sizes = [[32, 64, 128], [256], [512]]
        self.steps = [32, 64, 128]
        self.variance = [0.1, 0.2]
        self.confidence_threshold = 0.05
        self.nms_threshold = 0.3
        self.keep_top_k = 750

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

        result = []
        for identifier, scores, boxes, meta in zip(identifiers, batch_scores, batch_boxes, frame_meta):
            detections = {'labels': [], 'scores': [], 'x_mins': [], 'y_mins': [], 'x_maxs': [], 'y_maxs': []}
            image_info = meta.get("image_info")[0:2]

            # Prior boxes
            feature_maps = [[math.ceil(image_info[0] / step), math.ceil(image_info[1] / step)] for step in
                            self.steps]
            prior_data = self.prior_boxes(feature_maps, image_info)

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
                x_mins = x_mins[keep]
                y_mins = y_mins[keep]
                x_maxs = x_maxs[keep]
                y_maxs = y_maxs[keep]

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

class PersonVehicleDetectionAdapter(Adapter):
    __provider__ = 'person_vehicle_detection'
    predcition_types = (DetectionPrediction, )
    class_threshold_ = [0.0, 0.60, 0.85, 0.90, 0.70, 0.75, 0.80, 0.75]
    kMinObjectSize = 25

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'iou_threshold': NumberField(
                value_type=float, min_value=0, max_value=1, default=0.45, optional=True,
                description='Iou threshold for NMS'),
        })
        return parameters

    def configure(self):
        self.iou_threshold = self.get_value_from_config('iou_threshold')

    def process(self, raw, identifiers=None, frame_meta=None):
        result = []
        if isinstance(raw, dict):
            bbox_pred = raw['bbox_pred']
            proposals = raw['proposals']
            cls_score = raw['cls_score']
            img_height, img_width, _ = frame_meta[0]['image_size']
            props_map = self.output_to_proposals(bbox_pred, proposals, cls_score, img_width, img_height)
            pred_items = self.get_proposals(props_map)
            result.append(pred_items)
        else:
            for batch_index in range(len(identifiers)):
                bbox_pred = raw[batch_index]['bbox_pred']
                proposals = raw[batch_index]['proposals']
                cls_score = raw[batch_index]['cls_score']
                img_height, img_width, _ = frame_meta[batch_index]['image_size']
                props_map = self.output_to_proposals(bbox_pred, proposals, cls_score, img_width, img_height)
                pred_items = self.get_proposals(props_map)
                result.append(pred_items)
        return result

    def output_to_proposals(self, bbox_pred, proposals, cls_score, img_width, img_height):
        ww = img_width
        hh = img_height
        sx = 1.0 / (960 / img_width)
        sy = 1.0 / (540 / img_height)
        num_rois = 96
        num_classes = 8

        # merge scores - Car: 3, Truck: 7, Van: 6
        for r in range(num_rois):
            indices = [3, 6, 7]

            def get_cls_score(idx, x=r):
                return cls_score[x][idx]

            indices.sort(key=get_cls_score, reverse=True)

            sum_score = 0.0
            for it in indices:
                sum_score += cls_score[r][it]
                cls_score[r][it] = 0.0

            cls_score[r][indices[0]] = sum_score

        # proposals map [class id, rectangle]
        props_map = []

        # NMS in tensorpack way
        for c in range(1, num_classes):
            scores = []
            boxes = []

            for r in range(num_rois):
                # class score
                scores.append(cls_score[r][c])
                # regressed bbox
                reg = bbox_pred[r][4*c:]
                roi = proposals[r][1:]
                boxes.append(self.regress_scale_clip_bbox(ww, hh, sx, sy, roi, reg))

            # nms in a class
            kidx = self.nms_apply(c, scores, boxes)

            # final class and box
            p_map = []
            for i in kidx:
                p_map.append([c, scores[i], boxes[i]])  # class_id, score, box_rectangle

            props_map.append(p_map)

        return props_map

    def get_proposals(self, props_map):
        vehicles = []
        detections = {'labels': [], 'scores': [], 'x_mins': [], 'y_mins': [], 'x_maxs': [], 'y_maxs': []}

        for props in props_map:
            if len(props) == 0:
                continue

            for prop in props:
                class_id = prop[0]
                rect = prop[2]
                score = prop[1]
                if (rect[2] - rect[0]) < self.kMinObjectSize or (rect[3] - rect[1]) < self.kMinObjectSize:
                    continue

                gt_class_id = self.convert_classid_to_gt_type(class_id)

                # person
                if gt_class_id == 0:
                    detections['labels'].append(gt_class_id)
                    detections['scores'].append(score)
                    detections['x_mins'].append(rect[0])
                    detections['y_mins'].append(rect[1])
                    detections['x_maxs'].append(rect[2])
                    detections['y_maxs'].append(rect[3])

                # vehicle
                else:
                    vehicles.append(prop)

        # sort by score and NMS in 2 wheels, 4 heels
        vehicles = self.nms_apply_prop(vehicles)

        # Convert to detection representation
        for vehicle in vehicles:
            rect = vehicle[2]
            score = vehicle[1]
            detections['labels'].append(1)
            detections['scores'].append(score)
            detections['x_mins'].append(rect[0])
            detections['y_mins'].append(rect[1])
            detections['x_maxs'].append(rect[2])
            detections['y_maxs'].append(rect[3])

        return DetectionPrediction(
            labels=detections['labels'],
            scores=detections['scores'],
            x_mins=detections['x_mins'],
            y_mins=detections['y_mins'],
            x_maxs=detections['x_maxs'],
            y_maxs=detections['y_maxs']
        )

    @staticmethod
    def regress_scale_clip_bbox(ww, hh, sx, sy, roi, reg):
        cx = (roi[0] + roi[2]) * 0.5
        cy = (roi[1] + roi[3]) * 0.5
        aw = (roi[2] - roi[0])
        ah = (roi[3] - roi[1])

        cx += (reg[0] * 0.1 * aw)
        cy += (reg[1] * 0.1 * ah)
        aw *= (math.exp(reg[2] * 0.2) * 0.5)
        ah *= (math.exp(reg[3] * 0.2) * 0.5)

        res = [min(ww, max(0.0, (cx - aw) * sx)),
               min(hh, max(0.0, (cy - ah) * sy)),
               min(ww, max(0.0, (cx + aw) * sx)),
               min(hh, max(0.0, (cy + ah) * sy))]

        return res

    def nms_apply(self, class_id, scores, boxes):
        filtered = self.filter_by_score(self.class_threshold_[class_id], scores)
        idx = filtered[0]
        is_dead = filtered[1]

        # iou based filter(nms)
        keep_ids = []

        for i, _ in enumerate(idx):
            li = idx[i]
            if is_dead[li]:
                continue

            keep_ids.append(li)

            for j in range(i + 1, len(idx)):
                ri = idx[j]

                if is_dead[ri]:
                    continue

                iou = self.compute_iou(boxes[li], boxes[ri])
                if iou > self.iou_threshold:
                    is_dead[ri] = 1

        return keep_ids

    def nms_apply_prop(self, input_props):
        idx = []
        keep_ids = []
        is_dead = []

        for i in range(len(input_props)):
            idx.append(i)
            is_dead.append(False)

        def get_score(ii):
            val = input_props[ii][1]    # body_bbox_score
            return val

        idx.sort(key=get_score, reverse=True)

        for i in range(len(input_props)):
            li = idx[i]
            if is_dead[li]:
                continue

            keep_ids.append(li)

            for j in range(i+1, len(input_props)):
                ri = idx[j]

                if is_dead[ri]:
                    continue

                iou = self.compute_iou_prop(input_props[li], input_props[ri])

                if iou > self.iou_threshold:
                    is_dead[ri] = True

        output_props = []
        for it in keep_ids:
            if is_dead[it]:
                continue
            output_props.append(input_props[it])

        return output_props

    @staticmethod
    def filter_by_score(threshold, scores):
        # Argsort by score
        idx = []
        is_dead = []
        for i in range(len(scores)):
            idx.append(i)
            is_dead.append(True)

        def get_score(ii):
            val = scores[ii]
            return val

        idx.sort(key=get_score, reverse=True)

        # Score based filter
        r_idx = []
        for i in idx:
            if scores[i] < threshold:
                is_dead.append(True)
                continue
            r_idx.append(i)
            is_dead[i] = False

        return [r_idx, is_dead]

    @staticmethod
    def compute_iou(lhs, rhs):
        ix = min(lhs[2], rhs[2]) - max(lhs[0], rhs[0])
        iy = min(lhs[3], rhs[3]) - max(lhs[1], rhs[1])
        ai = max(0.0, ix) * max(0.0, iy)

        al = (lhs[2] - lhs[0]) * (lhs[3] - lhs[1])
        ar = (rhs[2] - rhs[0]) * (rhs[3] - rhs[1])
        uu = al + ar - ai   # Union

        return ai / max(uu, 0.0000001)

    @staticmethod
    def compute_iou_prop(lhs, rhs):
        lhs_box = lhs[2]
        rhs_box = rhs[2]
        return PersonVehicleDetectionAdapter.compute_iou(lhs_box, rhs_box)

    @staticmethod
    def convert_classid_to_gt_type(class_id):
        return 0 if class_id == 5 else 1
