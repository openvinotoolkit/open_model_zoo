"""
 Copyright (C) 2020-2022 Intel Corporation
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

from collections import namedtuple
import numpy as np

from .detection_model import DetectionModel
from .types import ListValue, NumericalValue
from .utils import Detection, clip_detections, nms, resize_image, INTERPOLATION_TYPES

DetectionBox = namedtuple('DetectionBox', ["x", "y", "w", "h"])

ANCHORS = {
    'YOLOV3': [10.0, 13.0, 16.0, 30.0, 33.0, 23.0,
               30.0, 61.0, 62.0, 45.0, 59.0, 119.0,
               116.0, 90.0, 156.0, 198.0, 373.0, 326.0],
    'YOLOV4': [12.0, 16.0, 19.0, 36.0, 40.0, 28.0,
               36.0, 75.0, 76.0, 55.0, 72.0, 146.0,
               142.0, 110.0, 192.0, 243.0, 459.0, 401.0],
    'YOLOV4-TINY': [10.0, 14.0, 23.0, 27.0, 37.0, 58.0,
                    81.0, 82.0, 135.0, 169.0, 344.0, 319.0],
    'YOLOF' : [16.0, 16.0, 32.0, 32.0, 64.0, 64.0,
               128.0, 128.0, 256.0, 256.0, 512.0, 512.0]
}

def permute_to_N_HWA_K(tensor, K, output_layout):
    """
    Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA), K)
    """
    assert tensor.ndim == 4, tensor.shape
    if output_layout == 'NHWC':
        tensor = tensor.transpose(0, 3, 1, 2)
    N, _, H, W = tensor.shape
    tensor = tensor.reshape(N, -1, K, H, W)
    tensor = tensor.transpose(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)
    return tensor

def sigmoid(x):
    return 1. / (1. + np.exp(-x))


class YOLO(DetectionModel):
    __model__ = 'YOLO'

    class Params:
        # Magic numbers are copied from yolo samples
        def __init__(self, param, sides):
            self.num = param.get('num', 3)
            self.coords = param.get('coord', 4)
            self.classes = param.get('classes', 80)
            self.bbox_size = self.coords + self.classes + 1
            self.sides = sides
            self.anchors = param.get('anchors', ANCHORS['YOLOV3'])

            self.use_input_size = False
            self.output_layout = 'NCHW'

            mask = param.get('mask', None)
            if mask:
                self.num = len(mask)

                masked_anchors = []
                for idx in mask:
                    masked_anchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
                self.anchors = masked_anchors

                self.use_input_size = True  # Weak way to determine but the only one.

    def __init__(self, model_adapter, configuration, preload=False):
        super().__init__(model_adapter, configuration, preload)
        self.is_tiny = len(self.outputs) == 2  # Weak way to distinguish between YOLOv4 and YOLOv4-tiny

        self._check_io_number(1, -1)

        self.yolo_layer_params = self._get_output_info()

    def _get_output_info(self):
        output_info = {}
        yolo_regions = self.model_adapter.operations_by_type('RegionYolo')
        for name, info in self.outputs.items():
            shape = info.shape
            if len(shape) == 2:
                # we use 32x32 cell as default, cause 1D tensor is V2 specific
                cx = self.w // 32
                cy = self.h // 32

                bboxes = shape[1] // (cx*cy)
                if self.w % 32 != 0 or self.h % 32 != 0 or shape[1] % (cx*cy) != 0:
                    self.raise_error('The cannot reshape 2D output tensor into 4D')
                shape = (shape[0], bboxes, cy, cx)
            meta = info.meta
            if info.type != 'RegionYolo' and yolo_regions:
                for region_name in yolo_regions:
                    if region_name in name:
                        meta = yolo_regions[region_name].meta
            params = self.Params(meta, shape[2:4])
            output_info[name] = (shape, params)
        return output_info

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'iou_threshold': NumericalValue(default_value=0.5, description="Threshold for NMS filtering"),
        })
        parameters['resize_type'].update_default_value('fit_to_window_letterbox')
        parameters['confidence_threshold'].update_default_value(0.5)
        return parameters

    def postprocess(self, outputs, meta):
        detections = self._parse_outputs(outputs, meta)
        detections = self._resize_detections(detections, meta)
        return detections

    def _parse_yolo_region(self, predictions, input_size, params):
        # ------------------------------------------ Extracting layer parameters ---------------------------------------
        objects = []
        size_normalizer = input_size if params.use_input_size else params.sides
        predictions = permute_to_N_HWA_K(predictions, params.bbox_size, params.output_layout)
        # ------------------------------------------- Parsing YOLO Region output ---------------------------------------
        for prediction in predictions:
            # Getting probabilities from raw outputs
            class_probabilities = self._get_probabilities(prediction, params.classes)

            # filter out the proposals with low confidence score
            keep_idxs = np.nonzero(class_probabilities > self.confidence_threshold)[0]
            class_probabilities = class_probabilities[keep_idxs]
            obj_indx = keep_idxs // params.classes
            class_idx = keep_idxs % params.classes

            for ind, obj_ind in enumerate(obj_indx):
                row, col, n = self._get_location(obj_ind, params.sides[0], params.num)

                # Process raw value to get absolute coordinates of boxes
                raw_box = self._get_raw_box(prediction, obj_ind)
                predicted_box = self._get_absolute_det_box(raw_box, row, col, params.anchors[2 * n:2 * n + 2],
                                                           params.sides, size_normalizer)

                # Define class_label and cofidence
                label = class_idx[ind]
                confidence = class_probabilities[ind]
                objects.append(Detection(predicted_box.x - predicted_box.w / 2,
                                         predicted_box.y - predicted_box.h / 2,
                                         predicted_box.x + predicted_box.w / 2,
                                         predicted_box.y + predicted_box.h / 2,
                                         confidence.item(), label.item()))

        return objects

    @staticmethod
    def _get_probabilities(prediction, classes):
        object_probabilities = prediction[:, 4].flatten()
        class_probabilities = prediction[:, 5:].flatten()
        class_probabilities *= np.repeat(object_probabilities, classes)
        return class_probabilities

    @staticmethod
    def _get_location(obj_ind, cells, num):
        row = obj_ind // (cells * num)
        col = (obj_ind - row * cells * num) // num
        n = (obj_ind - row * cells * num) % num
        return row, col, n

    @staticmethod
    def _get_raw_box(prediction, obj_ind):
        return DetectionBox(*prediction[obj_ind, :4])

    @staticmethod
    def _get_absolute_det_box(box, row, col, anchors, coord_normalizer, size_normalizer):
        x = (col + box.x) / coord_normalizer[1]
        y = (row + box.y) / coord_normalizer[0]
        width = np.exp(box.w) * anchors[0] / size_normalizer[0]
        height = np.exp(box.h) * anchors[1] / size_normalizer[1]

        return DetectionBox(x, y, width, height)

    @staticmethod
    def _filter(detections, iou_threshold):
        def iou(box_1, box_2):
            width_of_overlap_area = min(box_1.xmax, box_2.xmax) - max(box_1.xmin, box_2.xmin)
            height_of_overlap_area = min(box_1.ymax, box_2.ymax) - max(box_1.ymin, box_2.ymin)
            if width_of_overlap_area < 0 or height_of_overlap_area < 0:
                area_of_overlap = 0
            else:
                area_of_overlap = width_of_overlap_area * height_of_overlap_area
            box_1_area = (box_1.ymax - box_1.ymin) * (box_1.xmax - box_1.xmin)
            box_2_area = (box_2.ymax - box_2.ymin) * (box_2.xmax - box_2.xmin)
            area_of_union = box_1_area + box_2_area - area_of_overlap
            if area_of_union == 0:
                return 0
            return area_of_overlap / area_of_union

        detections = sorted(detections, key=lambda obj: obj.score, reverse=True)
        for i in range(len(detections)):
            if detections[i].score == 0:
                continue
            for j in range(i + 1, len(detections)):
                # We perform IOU only on objects of same class
                if detections[i].id != detections[j].id:
                    continue

                if iou(detections[i], detections[j]) > iou_threshold:
                    detections[j].score = 0

        return [det for det in detections if det.score > 0]

    def _parse_outputs(self, outputs, meta):
        detections = []
        for layer_name in self.yolo_layer_params.keys():
            out_blob = outputs[layer_name]
            layer_params = self.yolo_layer_params[layer_name]
            out_blob.shape = layer_params[0]
            detections += self._parse_yolo_region(out_blob, meta['resized_shape'], layer_params[1])

        detections = self._filter(detections, self.iou_threshold)
        return detections


class YoloV4(YOLO):
    __model__ = 'YOLOV4'

    class Params:
        def __init__(self, classes, num, sides, anchors, mask, layout):
            self.num = num
            self.coords = 4
            self.classes = classes
            self.bbox_size = self.coords + self.classes + 1
            self.sides = sides
            self.output_layout = layout
            masked_anchors = []
            for idx in mask:
                masked_anchors += [anchors[idx * 2], anchors[idx * 2 + 1]]
            self.anchors = masked_anchors
            self.use_input_size = True

    def __init__(self, model_adapter, configuration=None, preload=False):
        super().__init__(model_adapter, configuration, preload)

    def _get_output_info(self):
        if not self.anchors:
            self.anchors = ANCHORS['YOLOV4-TINY'] if self.is_tiny else ANCHORS['YOLOV4']
        if not self.masks:
            self.masks = [1, 2, 3, 3, 4, 5] if self.is_tiny else [0, 1, 2, 3, 4, 5, 6, 7, 8]

        outputs = sorted(self.outputs.items(), key=lambda x: x[1].shape[2], reverse=True)

        output_info = {}
        num = 3
        for i, (name, layer) in enumerate(outputs):
            shape = layer.shape
            if shape[2] == shape[3]:
                channels, sides, layout = shape[1], shape[2:4], 'NCHW'
            else:
                channels, sides, layout = shape[3], shape[1:3], 'NHWC'
            classes = channels // num - 5
            if channels % num != 0:
                self.raise_error("The output blob {} has wrong 2nd dimension".format(name))
            yolo_params = self.Params(classes, num, sides, self.anchors, self.masks[i*num : (i+1)*num], layout)
            output_info[name] = (shape, yolo_params)
        return output_info

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'anchors': ListValue(description="List of custom anchor values"),
            'masks': ListValue(description="List of mask, applied to anchors for each output layer"),
        })
        return parameters

    @staticmethod
    def _get_probabilities(prediction, classes):
        object_probabilities = sigmoid(prediction[:, 4].flatten())
        class_probabilities = sigmoid(prediction[:, 5:].flatten())
        class_probabilities *= np.repeat(object_probabilities, classes)
        return class_probabilities

    @staticmethod
    def _get_raw_box(prediction, obj_ind):
        bbox = prediction[obj_ind, :4]
        x, y = sigmoid(bbox[:2])
        width, height = bbox[2:]
        return DetectionBox(x, y, width, height)


class YOLOF(YOLO):
    __model__ = 'YOLOF'

    class Params:
        def __init__(self, classes, num, sides, anchors):
            self.num = num
            self.coords = 4
            self.classes = classes
            self.bbox_size = self.coords + self.classes
            self.sides = sides
            self.anchors = anchors
            self.output_layout = 'NCHW'
            self.use_input_size = True

    def __init__(self, model_adapter, configuration=None, preload=False):
        super().__init__(model_adapter, configuration, preload)

    def _get_output_info(self):
        anchors = ANCHORS['YOLOF']

        output_info = {}
        num = 6
        for name, layer in self.outputs.items():
            shape = layer.shape
            classes = shape[1] // num - 4
            yolo_params = self.Params(classes, num, shape[2:4], anchors)
            output_info[name] = (shape, yolo_params)
        return output_info

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters['resize_type'].update_default_value('standard')
        return parameters

    @staticmethod
    def _get_probabilities(prediction, classes):
        return sigmoid(prediction[:, 4:].flatten())

    @staticmethod
    def _get_absolute_det_box(box, row, col, anchors, coord_normalizer, size_normalizer):
        anchor_x = anchors[0] / size_normalizer[0]
        anchor_y = anchors[1] / size_normalizer[1]
        x = box.x * anchor_x + col / coord_normalizer[1]
        y = box.y * anchor_y + row / coord_normalizer[0]
        width = np.exp(box.w) * anchor_x
        height = np.exp(box.h) * anchor_y

        return DetectionBox(x, y, width, height)


class YOLOX(DetectionModel):
    __model__ = 'YOLOX'

    def __init__(self, model_adapter, configuration=None, preload=False):
        super().__init__(model_adapter, configuration, preload)
        self._check_io_number(1, 1)
        self.output_blob_name = next(iter(self.outputs))

        self.expanded_strides = []
        self.grids = []
        self.set_strides_grids()

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'iou_threshold': NumericalValue(default_value=0.65, description="Threshold for NMS filtering"),
        })
        parameters['confidence_threshold'].update_default_value(0.5)
        return parameters

    def preprocess(self, inputs):
        image = inputs
        resized_image = resize_image(image, (self.w, self.h), keep_aspect_ratio=True)

        padded_image = np.ones((self.h, self.w, 3), dtype=np.uint8) * 114
        padded_image[: resized_image.shape[0], : resized_image.shape[1]] = resized_image

        meta = {'original_shape': image.shape,
                'scale': min(self.w / image.shape[1], self.h / image.shape[0])}

        preprocessed_image = self.input_transform(padded_image)
        preprocessed_image = preprocessed_image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        preprocessed_image = preprocessed_image.reshape((self.n, self.c, self.h, self.w))

        dict_inputs = {self.image_blob_name: preprocessed_image}
        return dict_inputs, meta

    def postprocess(self, outputs, meta):
        output = outputs[self.output_blob_name][0]

        if np.size(self.expanded_strides) != 0 and np.size(self.grids) != 0:
            output[..., :2] = (output[..., :2] + self.grids) * self.expanded_strides
            output[..., 2:4] = np.exp(output[..., 2:4]) * self.expanded_strides

        valid_predictions = output[output[..., 4] > self.confidence_threshold]
        valid_predictions[:, 5:] *= valid_predictions[:, 4:5]

        boxes = self.xywh2xyxy(valid_predictions[:, :4]) / meta['scale']
        i, j = (valid_predictions[:, 5:] > self.confidence_threshold).nonzero()
        x_mins, y_mins, x_maxs, y_maxs = boxes[i].T
        scores = valid_predictions[i, j + 5]

        keep_nms = nms(x_mins, y_mins, x_maxs, y_maxs, scores, self.iou_threshold, include_boundaries=True)

        detections = [Detection(*det) for det in zip(x_mins[keep_nms], y_mins[keep_nms], x_maxs[keep_nms],
                                                     y_maxs[keep_nms], scores[keep_nms], j[keep_nms])]
        return clip_detections(detections, meta['original_shape'])

    def set_strides_grids(self):
        grids = []
        expanded_strides = []

        strides = [8, 16, 32]

        hsizes = [self.h // stride for stride in strides]
        wsizes = [self.w // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        self.grids = np.concatenate(grids, 1)
        self.expanded_strides = np.concatenate(expanded_strides, 1)

    @staticmethod
    def xywh2xyxy(x):
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y


class YoloV3ONNX(DetectionModel):
    __model__ = 'YOLOv3-ONNX'

    def __init__(self, model_adapter, configuration=None, preload=False):
        super().__init__(model_adapter, configuration, preload)
        self.image_info_blob_name = self.image_info_blob_names[0] if len(self.image_info_blob_names) == 1 else None
        self._check_io_number(2, 3)
        self.classes = 80
        self.bboxes_blob_name, self.scores_blob_name, self.indices_blob_name = self._get_outputs()

    def _get_outputs(self):
        bboxes_blob_name = None
        scores_blob_name = None
        indices_blob_name = None
        for name, layer in self.outputs.items():
            if layer.shape[-1] == 3:
                indices_blob_name = name
            elif layer.shape[2] == 4:
                bboxes_blob_name = name
            elif layer.shape[1] == self.classes:
                scores_blob_name = name
            else:
                self.raise_error("Expected shapes [:,:,4], [:,{},:] and [:,3] for outputs, but got {}, {} and {}"
                    .format(self.classes, *[output.shape for output in self.outputs.values()]))
        if self.outputs[bboxes_blob_name].shape[1] != self.outputs[scores_blob_name].shape[2]:
            self.raise_error("Expected the same dimension for boxes and scores, but got {} and {}".format(
                self.outputs[bboxes_blob_name].shape[1], self.outputs[scores_blob_name].shape[2]))
        return bboxes_blob_name, scores_blob_name, indices_blob_name

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters['resize_type'].update_default_value('fit_to_window_letterbox')
        parameters['confidence_threshold'].update_default_value(0.5)
        return parameters

    def preprocess(self, inputs):
        image = inputs
        meta = {'original_shape': image.shape}
        resized_image = self.resize(image, (self.w, self.h), interpolation=INTERPOLATION_TYPES['CUBIC'])
        meta.update({'resized_shape': resized_image.shape})
        resized_image = self._change_layout(resized_image)
        dict_inputs = {
            self.image_blob_name: resized_image,
            self.image_info_blob_name: [[image.shape[0], image.shape[1]]]
        }
        return dict_inputs, meta

    def postprocess(self, outputs, meta):
        detections = self._parse_outputs(outputs)
        detections = clip_detections(detections, meta['original_shape'])
        return detections

    def _parse_outputs(self, outputs):
        boxes = outputs[self.bboxes_blob_name][0]
        scores = outputs[self.scores_blob_name][0]
        indices = outputs[self.indices_blob_name] if len(
            outputs[self.indices_blob_name].shape) == 2 else outputs[self.indices_blob_name][0]

        out_boxes, out_scores, out_classes = [], [], []
        for idx_ in indices:
            if idx_[0] == -1:
                break
            out_classes.append(idx_[1])
            out_scores.append(scores[tuple(idx_[1:])])
            out_boxes.append(boxes[idx_[2]])
        transposed_boxes = np.array(out_boxes).T if out_boxes else ([], [], [], [])
        mask = np.array(out_scores) > self.confidence_threshold

        if mask.size == 0:
            return []

        out_classes, out_scores, transposed_boxes = (np.array(out_classes)[mask], np.array(out_scores)[mask],
                                                     transposed_boxes[:, mask])

        x_mins = transposed_boxes[1]
        y_mins = transposed_boxes[0]
        x_maxs = transposed_boxes[3]
        y_maxs = transposed_boxes[2]

        detections = [Detection(*det) for det in zip(x_mins, y_mins, x_maxs,
                                                     y_maxs, out_scores, out_classes)]

        return detections
