"""
 Copyright (C) 2020-2021 Intel Corporation

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
import ngraph

from .detection_model import DetectionModel
from .utils import Detection

ANCHORS = {
    'YOLOV3': [10.0, 13.0, 16.0, 30.0, 33.0, 23.0,
               30.0, 61.0, 62.0, 45.0, 59.0, 119.0,
               116.0, 90.0, 156.0, 198.0, 373.0, 326.0],
    'YOLOV4': [12.0, 16.0, 19.0, 36.0, 40.0, 28.0,
               36.0, 75.0, 76.0, 55.0, 72.0, 146.0,
               142.0, 110.0, 192.0, 243.0, 459.0, 401.0],
    'YOLOV4-TINY': [10.0, 14.0, 23.0, 27.0, 37.0, 58.0,
                    81.0, 82.0, 135.0, 169.0, 344.0, 319.0]
}

def permute_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA), K)
    """
    assert tensor.ndim == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.reshape(N, -1, K, H, W)
    tensor = tensor.transpose(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)
    return tensor

class YOLO(DetectionModel):
    class Params:
        # Magic numbers are copied from yolo samples
        def __init__(self, param, sides):
            self.num = param.get('num', 3)
            self.coords = param.get('coord', 4)
            self.classes = param.get('classes', 80)
            self.sides = sides
            self.anchors = param.get('anchors', ANCHORS['YOLOV3'])

            self.isYoloV3 = False

            mask = param.get('mask', None)
            if mask:
                self.num = len(mask)

                masked_anchors = []
                for idx in mask:
                    masked_anchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
                self.anchors = masked_anchors

                self.isYoloV3 = True  # Weak way to determine but the only one.

    def __init__(self, ie, model_path, input_transform=None, resize_type='letterbox',
                 labels=None, threshold=0.5, iou_threshold=0.5):
        super().__init__(ie, model_path, input_transform=input_transform, resize_type=resize_type,
                         labels=labels, threshold=threshold, iou_threshold=iou_threshold)
        self.is_tiny = self.net.name.lower().find('tiny') != -1  # Weak way to distinguish between YOLOv4 and YOLOv4-tiny

        if len(self.inputs) != 1:
            raise RuntimeError(f"The YOLO model family expects only one input blob but {len(self.inputs)} found:\n"
                               f"{','.join(*self.inputs)}")

        if self.net.input_info[self.image_blob_name].input_data.shape[1] == 3:
            self.n, self.c, self.h, self.w = self.net.input_info[self.image_blob_name].input_data.shape
            self.image_layout = 'NCHW'
        else:
            self.n, self.h, self.w, self.c = self.net.input_info[self.image_blob_name].input_data.shape
            self.image_layout = 'NHWС'

        self.yolo_layer_params = self._get_output_info()

    def _get_output_info(self):
        def get_parent(node):
            return node.inputs()[0].get_source_output().get_node()
        ng_func = ngraph.function_from_cnn(self.net)
        output_info = {}
        for node in ng_func.get_ordered_ops():
            layer_name = node.get_friendly_name()
            if layer_name not in self.net.outputs:
                continue
            shape = list(get_parent(node).shape)
            yolo_params = self.Params(node._get_attributes(), shape[2:4])
            output_info[layer_name] = (shape, yolo_params)
        return output_info

    def postprocess(self, outputs, meta):
        detections = self._parse_outputs(outputs, meta)
        detections = self._resize_detections(detections, meta)
        return detections

    @staticmethod
    def _parse_yolo_region(predictions, input_size, params, threshold):
        # ------------------------------------------ Extracting layer parameters ---------------------------------------
        objects = []
        size_normalizer = input_size if params.isYoloV3 else params.sides
        bbox_size = params.coords + 1 + params.classes
        predictions = permute_to_N_HWA_K(predictions, bbox_size)
        # ------------------------------------------- Parsing YOLO Region output ---------------------------------------
        for prediction in predictions:
            # Getting probabilities from raw outputs
            object_probabilities = prediction[:, 4].flatten()
            class_probabilities = prediction[:, 5:].flatten()
            class_probabilities *= np.repeat(object_probabilities, params.classes)

            # filter out the proposals with low confidence score
            keep_idxs = np.nonzero(class_probabilities > threshold)[0]
            class_probabilities = class_probabilities[keep_idxs]
            obj_indx = keep_idxs // params.classes
            class_idx = keep_idxs % params.classes

            for ind, obj_ind in enumerate(obj_indx):
                x, y, width, height = prediction[:, :4][obj_ind]

                row = obj_ind // (params.sides[0] * params.num)
                col = (obj_ind - row * params.sides[0] * params.num) // params.num
                n = (obj_ind - row * params.sides[0] * params.num) % params.num

                # Process raw value to get absolute coordinates of boxes
                x = (col + x) / params.sides[1]
                y = (row + y) / params.sides[0]
                # Value for exp is very big number in some cases so following construction is using here
                try:
                    width = np.exp(width)
                    height = np.exp(height)
                except OverflowError:
                    continue
                # Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
                width = width * params.anchors[2 * n] / size_normalizer[0]
                height = height * params.anchors[2 * n + 1] / size_normalizer[1]

                # Define class_label and cofidence
                label = class_idx[ind]
                confidence = class_probabilities[ind]
                objects.append(Detection(x - width / 2, y - height / 2, x + width / 2, y + height / 2,
                                         confidence.item(), label.item()))

        return objects

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
            detections += self._parse_yolo_region(out_blob, meta['resized_shape'], layer_params[1], self.threshold)

        detections = self._filter(detections, self.iou_threshold)
        return detections


class YoloV4(YOLO):
    class Params:
        def __init__(self, classes, num, sides, anchors, mask):
            self.num = num
            self.coords = 4
            self.classes = classes
            self.sides = sides
            masked_anchors = []
            for idx in mask:
                masked_anchors += [anchors[idx * 2], anchors[idx * 2 + 1]]
            self.anchors = masked_anchors

    def __init__(self, ie, model_path, input_transform=None, resize_type='letterbox',
                 labels=None, threshold=0.5, iou_threshold=0.5,
                 anchors=None, masks=None):
        self.anchors = anchors
        self.masks = masks
        super().__init__(ie, model_path, input_transform=input_transform, resize_type=resize_type,
                         labels=labels, threshold=threshold, iou_threshold=iou_threshold)

    def _get_output_info(self):
        if not self.anchors:
            self.anchors = ANCHORS['YOLOV4-TINY'] if self.is_tiny else ANCHORS['YOLOV4']
        if not self.masks:
            self.masks = [1, 2, 3, 3, 4, 5] if self.is_tiny else [0, 1, 2, 3, 4, 5, 6, 7, 8]

        outputs = sorted(self.net.outputs.items(), key=lambda x: x[1].shape[2], reverse=True)

        output_info = {}
        num = 3
        for i, (name, layer) in enumerate(outputs):
            shape = layer.shape
            classes = shape[1] // num - 5
            if shape[1] % num != 0:
                raise RuntimeError("The output blob {} has wrong 2nd dimension".format(name))
            yolo_params = self.Params(classes, num, shape[2:4], self.anchors, self.masks[i*num : (i+1)*num])
            output_info[name] = (shape, yolo_params)
        return output_info

    @staticmethod
    def _parse_yolo_region(predictions, input_size, params, threshold):
        def sigmoid(x):
            return 1. / (1. + np.exp(-x))

        objects = []
        bbox_size = params.coords + 1 + params.classes
        predictions = permute_to_N_HWA_K(predictions, bbox_size)
        # ------------------------------------------- Parsing YOLO Region output ---------------------------------------
        for prediction in predictions:
            # Getting probabilities from raw outputs
            object_probabilities = sigmoid(prediction[:, 4].flatten())
            class_probabilities = sigmoid(prediction[:, 5:].flatten())
            class_probabilities *= np.repeat(object_probabilities, params.classes)

            # filter out the proposals with low confidence score
            keep_idxs = np.nonzero(class_probabilities > threshold)[0]
            class_probabilities = class_probabilities[keep_idxs]
            obj_indx = keep_idxs // params.classes
            class_idx = keep_idxs % params.classes

            for ind, obj_ind in enumerate(obj_indx):
                bbox = prediction[:, :4][obj_ind]
                x, y = sigmoid(bbox[:2])
                width, height = bbox[2:]

                row = obj_ind // (params.sides[0] * params.num)
                col = (obj_ind - row * params.sides[0] * params.num) // params.num
                n = (obj_ind - row * params.sides[0] * params.num) % params.num

                # Process raw value to get absolute coordinates of boxes
                x = (col + x) / params.sides[1]
                y = (row + y) / params.sides[0]
                # Value for exp is very big number in some cases so following construction is using here
                try:
                    width = np.exp(width)
                    height = np.exp(height)
                except OverflowError:
                    continue
                width = width * params.anchors[2 * n] / input_size[0]
                height = height * params.anchors[2 * n + 1] / input_size[1]

                # Define class_label and cofidence
                label = class_idx[ind]
                confidence = class_probabilities[ind]
                objects.append(Detection(x - width / 2, y - height / 2, x + width / 2, y + height / 2,
                                         confidence.item(), label.item()))

        return objects


class YOLOX(Model):
    def __init__(self, ie, model_path, labels=None, threshold=0.5):
        super().__init__(ie, model_path)

        assert len(self.net.input_info) == 1, "Expected 1 input blob"
        self.image_blob_name = next(iter(self.net.input_info))

        assert len(self.net.outputs) == 1, "Expected 1 output blob"
        self.output_blob_name = next(iter(self.net.outputs))

        self.n, self.c, self.h, self.w = self.net.input_info[self.image_blob_name].input_data.shape
        assert self.c == 3, "Expected 3-channel input"

        if isinstance(labels, (list, tuple)):
            self.labels = labels
        else:
            self.labels = load_labels(labels) if labels else None

        self.threshold = threshold
        self.nms_threshold = 0.65
        self.expanded_strides = []
        self.grids = []
        self.set_strides_grids()

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

        valid_predictions = output[output[..., 4] > self.threshold]
        valid_predictions[:, 5:] *= valid_predictions[:, 4:5]

        boxes = self.xywh2xyxy(valid_predictions[:, :4]) / meta['scale']
        i, j = (valid_predictions[:, 5:] > self.threshold).nonzero()
        x_mins, y_mins, x_maxs, y_maxs = boxes[i].T
        scores = valid_predictions[i, j + 5]

        keep_nms = nms(x_mins, y_mins, x_maxs, y_maxs, scores, self.nms_threshold, include_boundaries=True)

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
