import math

import numpy as np

from ..adapters import Adapter
from ..config import NumberField, StringField, ConfigValidator, ListField, ConfigError
from ..representation import DetectionPrediction
from ..topology_types import YoloV1Tiny, YoloV2, YoloV2Tiny, YoloV3, YoloV3Tiny
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
                            "{}".format(', '.join(YoloV2Adapter.PRECOMPUTED_ANCHORS))
            ),
            'cells': NumberField(
                value_type=int, optional=True, min_value=1, default=13,
                description="Number of cells across width and height"
            )
        })
        return parameters

    def validate_config(self):
        super().validate_config(on_extra_argument=ConfigValidator.WARN_ON_EXTRA_ARGUMENT)

    def configure(self):
        self.classes = self.get_value_from_config('classes')
        self.coords = self.get_value_from_config('coords')
        self.num = self.get_value_from_config('num')
        self.anchors = get_or_parse_value(self.get_value_from_config('anchors'), YoloV2Adapter.PRECOMPUTED_ANCHORS)
        self.cells = self.get_value_from_config('cells')

    def process(self, raw, identifiers=None, frame_meta=None):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
        Returns:
            list of DetectionPrediction objects
        """
        predictions = self._extract_predictions(raw, frame_meta)[self.output_blob]

        cells_x, cells_y = self.cells, self.cells

        result = []
        for identifier, prediction in zip(identifiers, predictions):
            labels, scores, x_mins, y_mins, x_maxs, y_maxs = [], [], [], [], [], []
            if len(np.shape(prediction)) == 3:
                prediction = prediction.flatten()
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
            cells_y, cells_x = prediction.shape[1:]
            anchors = self.masked_anchors[layer_id] if self.masked_anchors else self.anchors
            num = len(anchors) // 2 if self.masked_anchors else self.num
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
            nchw_layout = input_shape[1] == 3
            self.input_width = input_shape[3 if nchw_layout else 2]
            self.input_height = input_shape[2 if nchw_layout else 1]

            for layer_id, p in enumerate(prediction):
                parse_yolo_v3_results(p, self.threshold, self.input_width, self.input_height, detections, layer_id)

            result.append(DetectionPrediction(
                identifier, detections['labels'], detections['scores'], detections['x_mins'], detections['y_mins'],
                detections['x_maxs'], detections['y_maxs']
            ))

        return result
