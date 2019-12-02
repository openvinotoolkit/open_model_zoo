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

import numpy as np

from ..topology_types import YoloV1Tiny, YoloV2, YoloV2Tiny, YoloV3, YoloV3Tiny, SSD, FasterRCNN
from ..adapters import Adapter
from ..config import ConfigValidator, NumberField, StringField, ListField, ConfigError
from ..postprocessor.nms import NMS
from ..representation import DetectionPrediction, ContainerPrediction
from ..utils import get_or_parse_value


class TinyYOLOv1Adapter(Adapter):
    """
    Class for converting output of Tiny YOLO v1 model to DetectionPrediction representation
    """
    __provider__ = 'tiny_yolo_v1'
    prediction_types = (DetectionPrediction, )
    topology_types = (YoloV1Tiny, )

    def process(self, raw, identifiers=None, frame_meta=None):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
        Returns:
             list of DetectionPrediction objects
        """
        prediction = self._extract_predictions(raw, frame_meta)[self.output_blob]

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

                labels.append(label)
                scores.append(score)
                x_mins.append(box[0] - box[2] / 2.0)
                y_mins.append(box[1] - box[3] / 2.0)
                x_maxs.append(box[0] + box[2] / 2.0)
                y_maxs.append(box[1] + box[3] / 2.0)

            result.append(DetectionPrediction(identifier, labels, scores, x_mins, y_mins, x_maxs, y_maxs))

        return result


def entry_index(w, h, n_coords, n_classes, pos, entry):
    row = pos // (w * h)
    col = pos % (w * h)
    return row * w * h * (n_classes + n_coords + 1) + entry * w * h + col


class YoloV2Adapter(Adapter):
    """
    Class for converting output of YOLO v2 family models to DetectionPrediction representation
    """
    __provider__ = 'yolo_v2'
    prediction_types = (DetectionPrediction, )
    topology_types = (YoloV2, YoloV2Tiny, )

    PRECOMPUTED_ANCHORS = {
        'yolo_v2': [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071],
        'tiny_yolo_v2': [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
    }

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'classes': NumberField(
                value_type=int, optional=True, min_value=1, default=20, description="Number of detection classes."
            ),
            'coords': NumberField(
                value_type=int, optional=True, min_value=1, default=4, description="Number of bbox coordinates."
            ),
            'num': NumberField(
                value_type=int, optional=True, min_value=1, default=5,
                description="Num parameter from DarkNet configuration file."
            ),
            'anchors': StringField(
                optional=True, choices=YoloV2Adapter.PRECOMPUTED_ANCHORS,
                allow_own_choice=True, default='yolo_v2',
                description="Anchor values provided as comma-separated list or one of precomputed: "
                            "{}".format(', '.join(YoloV2Adapter.PRECOMPUTED_ANCHORS)))
        })
        return parameters

    def validate_config(self):
        super().validate_config(on_extra_argument=ConfigValidator.WARN_ON_EXTRA_ARGUMENT)

    def configure(self):
        self.classes = self.get_value_from_config('classes')
        self.coords = self.get_value_from_config('coords')
        self.num = self.get_value_from_config('num')
        self.anchors = get_or_parse_value(self.get_value_from_config('anchors'), YoloV2Adapter.PRECOMPUTED_ANCHORS)

    def process(self, raw, identifiers=None, frame_meta=None):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
        Returns:
            list of DetectionPrediction objects
        """
        predictions = self._extract_predictions(raw, frame_meta)[self.output_blob]

        cells_x, cells_y = 13, 13

        result = []
        for identifier, prediction in zip(identifiers, predictions):
            labels, scores, x_mins, y_mins, x_maxs, y_maxs = [], [], [], [], [], []
            for y, x, n in np.ndindex((cells_y, cells_x, self.num)):
                index = n * cells_y * cells_x + y * cells_x + x

                box_index = entry_index(cells_x, cells_y, self.coords, self.classes, index, 0)
                obj_index = entry_index(cells_x, cells_y, self.coords, self.classes, index, self.coords)

                scale = prediction[obj_index]

                box = [
                    (x + prediction[box_index + 0 * (cells_y * cells_x)]) / cells_x,
                    (y + prediction[box_index + 1 * (cells_y * cells_x)]) / cells_y,
                    np.exp(prediction[box_index + 2 * (cells_y * cells_x)]) * self.anchors[2 * n + 0] / cells_x,
                    np.exp(prediction[box_index + 3 * (cells_y * cells_x)]) * self.anchors[2 * n + 1] / cells_y
                ]

                classes_prob = np.empty(self.classes)
                for cls in range(self.classes):
                    cls_index = entry_index(cells_x, cells_y, self.coords, self.classes, index, self.coords + 1 + cls)
                    classes_prob[cls] = prediction[cls_index]

                classes_prob = classes_prob * scale

                label = np.argmax(classes_prob)

                labels.append(label)
                scores.append(classes_prob[label])
                x_mins.append(box[0] - box[2] / 2.0)
                y_mins.append(box[1] - box[3] / 2.0)
                x_maxs.append(box[0] + box[2] / 2.0)
                y_maxs.append(box[1] + box[3] / 2.0)

            result.append(DetectionPrediction(identifier, labels, scores, x_mins, y_mins, x_maxs, y_maxs))

        return result


class YoloV3Adapter(Adapter):
    """
    Class for converting output of YOLO v3 family models to DetectionPrediction representation
    """
    __provider__ = 'yolo_v3'
    prediction_types = (DetectionPrediction, )
    topology_types = (YoloV3, YoloV3Tiny, )

    PRECOMPUTED_ANCHORS = {
        'yolo_v3': [
            10.0, 13.0,
            16.0, 30.0,
            33.0, 23.0,
            30.0, 61.0,
            62.0, 45.0,
            59.0, 119.0,
            116.0, 90.0,
            156.0, 198.0,
            373.0, 326.0
        ],
        'tiny_yolo_v3': [
            10.0, 14.0,
            23.0, 27.0,
            37.0, 58.0,
            81.0, 82.0,
            135.0, 169.0,
            344.0, 319.0
        ]
    }

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'classes': NumberField(
                value_type=int, optional=True, min_value=1, default=80, description="Number of detection classes."
            ),
            'coords': NumberField(
                value_type=int, optional=True, min_value=1, default=4, description="Number of bbox coordinates."
            ),
            'num': NumberField(
                value_type=int, optional=True, min_value=1, default=3,
                description="Num parameter from DarkNet configuration file."
            ),
            'anchors': StringField(
                optional=True, choices=YoloV3Adapter.PRECOMPUTED_ANCHORS.keys(), allow_own_choice=True,
                default='yolo_v3',
                description="Anchor values provided as comma-separated list or one of precomputed: "
                            "{}.".format(', '.join(YoloV3Adapter.PRECOMPUTED_ANCHORS.keys()))),
            'threshold': NumberField(value_type=float, optional=True, min_value=0, default=0.001,
                                     description="Minimal objectiveness score value for valid detections."),
            'outputs': ListField(
                optional=True, default=[],
                description="The list of output layers names (optional),"
                            " if specified there should be exactly 3 output layers provided."
            ),
            'anchor_masks': ListField(optional=True, description='per layer used anchors mask')
        })

        return parameters

    def validate_config(self):
        super().validate_config(on_extra_argument=ConfigValidator.WARN_ON_EXTRA_ARGUMENT)

    def configure(self):
        self.classes = self.get_value_from_config('classes')
        self.coords = self.get_value_from_config('coords')
        self.num = self.get_value_from_config('num')
        self.anchors = get_or_parse_value(self.get_value_from_config('anchors'), YoloV3Adapter.PRECOMPUTED_ANCHORS)
        self.threshold = self.get_value_from_config('threshold')
        self.outputs = self.get_value_from_config('outputs')
        anchor_masks = self.get_value_from_config('anchor_masks')
        self.masked_anchors = None
        if anchor_masks is not None:
            per_layer_anchors = []
            for layer_mask in anchor_masks:
                layer_anchors = []
                for idx in layer_mask:
                    layer_anchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
                per_layer_anchors.append(layer_anchors)
            self.masked_anchors = per_layer_anchors

    def process(self, raw, identifiers=None, frame_meta=None):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
        Returns:
            list of DetectionPrediction objects
        """

        def get_anchors_offset(x, num, anchors):
            return int((num * 2) * (len(anchors) / (num * 2) - 1 - math.log2(x / 13)))

        def parse_yolo_v3_results(prediction, threshold, w, h, det, layer_id):
            cells_x, cells_y = prediction.shape[1:]
            anchors = self.masked_anchors[layer_id] if self.masked_anchors else self.anchors
            num = len(anchors) // 2
            prediction = prediction.flatten()
            for y, x, n in np.ndindex((cells_y, cells_x, num)):
                index = n * cells_y * cells_x + y * cells_x + x
                anchors_offset = get_anchors_offset(cells_x, num, anchors) if not self.masked_anchors else 0

                box_index = entry_index(cells_x, cells_y, self.coords, self.classes, index, 0)
                obj_index = entry_index(cells_x, cells_y, self.coords, self.classes, index, self.coords)

                scale = prediction[obj_index]
                if scale < threshold:
                    continue

                box = [
                    (x + prediction[box_index + 0 * (cells_y * cells_x)]) / cells_x,
                    (y + prediction[box_index + 1 * (cells_y * cells_x)]) / cells_y,
                    np.exp(prediction[box_index + 2 * (cells_y * cells_x)]) * anchors[anchors_offset + 2 * n + 0] / w,
                    np.exp(prediction[box_index + 3 * (cells_y * cells_x)]) * anchors[anchors_offset + 2 * n + 1] / h
                ]

                classes_prob = np.empty(self.classes)
                for cls in range(self.classes):
                    cls_index = entry_index(cells_x, cells_y, self.coords, self.classes, index,
                                            self.coords + 1 + cls)
                    classes_prob[cls] = prediction[cls_index] * scale

                    det['labels'].append(cls)
                    det['scores'].append(classes_prob[cls])
                    det['x_mins'].append(box[0] - box[2] / 2.0)
                    det['y_mins'].append(box[1] - box[3] / 2.0)
                    det['x_maxs'].append(box[0] + box[2] / 2.0)
                    det['y_maxs'].append(box[1] + box[3] / 2.0)

            return det

        result = []

        raw_outputs = self._extract_predictions(raw, frame_meta)

        if self.outputs:
            outputs = self.outputs
        else:
            outputs = raw_outputs.keys()

        if self.masked_anchors and len(self.masked_anchors) != len(outputs):
            raise ConfigError('anchor mask should be specified for all output layers')
        batch = len(identifiers)
        predictions = [[] for _ in range(batch)]
        for blob in outputs:
            for b in range(batch):
                predictions[b].append(raw_outputs[blob][b])

        for identifier, prediction, meta in zip(identifiers, predictions, frame_meta):
            detections = {'labels': [], 'scores': [], 'x_mins': [], 'y_mins': [], 'x_maxs': [], 'y_maxs': []}
            input_shape = list(meta.get('input_shape', {'data': (1, 3, 416, 416)}).values())[0]
            self.input_width = input_shape[3]
            self.input_height = input_shape[2]

            for layer_id, p in enumerate(prediction):
                parse_yolo_v3_results(p, self.threshold, self.input_width, self.input_height, detections, layer_id)

            result.append(DetectionPrediction(
                identifier, detections['labels'], detections['scores'], detections['x_mins'], detections['y_mins'],
                detections['x_maxs'], detections['y_maxs']
            ))

        return result


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
            'keep_top_k': NumberField(optional=True, value_type=int, default=200, description="Keep top K.")
        })

        return parameters

    def configure(self):
        self.scores_out = self.get_value_from_config('scores_out')
        self.boxes_out = self.get_value_from_config('boxes_out')
        self.confidence_threshold = self.get_value_from_config('confidence_threshold')
        self.nms_threshold = self.get_value_from_config('nms_threshold')
        self.keep_top_k = self.get_value_from_config('keep_top_k')

        # Set default values according to:
        # https://github.com/mlperf/inference/tree/master/cloud/single_stage_detector
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.feat_size = [[50, 50], [25, 25], [13, 13], [7, 7], [3, 3], [3, 3]]
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

        result = []
        for identifier, scores, boxes, meta in zip(identifiers, batch_scores, batch_boxes, frame_meta):
            detections = {'labels': [], 'scores': [], 'x_mins': [], 'y_mins': [], 'x_maxs': [], 'y_maxs': []}
            image_info = meta.get("image_info")[0:2]

            # Default boxes
            dboxes = self.default_boxes(image_info, self.feat_size, self.scales, self.aspect_ratios)

            # Scores
            scores = np.transpose(scores)
            scores = self.softmax(scores, axis=1)

            # Boxes
            boxes = np.transpose(boxes)
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
    Class for converting output of MxNet SSD model to DetectionPrediction representation
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
