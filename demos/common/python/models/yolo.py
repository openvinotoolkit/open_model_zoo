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

from .model import Model
from .utils import Detection, resize_image, resize_image_letterbox, load_labels, clip_detections

ANCHORS = {
    'YOLOV3': [10.0, 13.0, 16.0, 30.0, 33.0, 23.0,
               30.0, 61.0, 62.0, 45.0, 59.0, 119.0,
               116.0, 90.0, 156.0, 198.0, 373.0, 326.0],
    'YOLOV4': [12.0, 16.0, 19.0, 36.0, 40.0, 28.0,
               36.0, 75.0, 76.0, 55.0, 72.0, 146.0,
               142.0, 110.0, 192.0, 243.0, 459.0, 401.0],
    'YOLOV4-TINY': [10.0, 14.0, 23.0, 27.0, 37.0, 58.0,
                    81.0, 82.0, 135.0, 169.0, 344.0, 319.0],
    'YOLOF': [16.0, 16.0, 32.0, 32.0, 64.0, 64.0,
              128.0, 128.0, 256.0, 256.0, 512.0, 512.0]
}

class YOLO(Model):
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

    def __init__(self, ie, model_path, labels=None, keep_aspect_ratio=False, threshold=0.5, iou_threshold=0.5):
        super().__init__(ie, model_path)

        self.is_tiny = self.net.name.lower().find('tiny') != -1  # Weak way to distinguish between YOLOv4 and YOLOv4-tiny

        if isinstance(labels, (list, tuple)):
            self.labels = labels
        else:
            self.labels = load_labels(labels) if labels else None

        self.threshold = threshold
        self.iou_threshold = iou_threshold

        self.keep_aspect_ratio = keep_aspect_ratio
        self.resize_image = resize_image_letterbox if self.keep_aspect_ratio else resize_image

        assert len(self.net.input_info) == 1, "Expected 1 input blob"
        self.image_blob_name = next(iter(self.net.input_info))
        if self.net.input_info[self.image_blob_name].input_data.shape[1] == 3:
            self.n, self.c, self.h, self.w = self.net.input_info[self.image_blob_name].input_data.shape
            self.nchw_shape = True
        else:
            self.n, self.h, self.w, self.c = self.net.input_info[self.image_blob_name].input_data.shape
            self.nchw_shape = False

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

    def preprocess(self, inputs):
        image = inputs

        resized_image = self.resize_image(image, (self.w, self.h))
        meta = {'original_shape': image.shape,
                'resized_shape': resized_image.shape}
        if self.nchw_shape:
            resized_image = resized_image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            resized_image = resized_image.reshape((self.n, self.c, self.h, self.w))

        else:
            resized_image = resized_image.reshape((self.n, self.h, self.w, self.c))

        dict_inputs = {self.image_blob_name: resized_image}
        return dict_inputs, meta

    @staticmethod
    def _parse_yolo_region(predictions, input_size, params, threshold, multiple_labels=True):
        # ------------------------------------------ Extracting layer parameters ---------------------------------------
        objects = []
        size_normalizer = input_size if params.isYoloV3 else params.sides
        bbox_size = params.coords + 1 + params.classes
        # ------------------------------------------- Parsing YOLO Region output ---------------------------------------
        for row, col, n in np.ndindex(params.sides[0], params.sides[1], params.num):
            # Getting raw values for each detection bounding bFox
            bbox = predictions[0, n * bbox_size:(n + 1) * bbox_size, row, col]
            x, y, width, height, object_probability = bbox[:5]
            class_probabilities = bbox[5:]
            if object_probability < threshold:
                continue
            # Process raw value
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

            if multiple_labels:
                for class_id, class_probability in enumerate(class_probabilities):
                    confidence = object_probability * class_probability
                    if confidence > threshold:
                        objects.append(Detection(x - width / 2, y - height / 2, x + width / 2, y + height / 2,
                                                 confidence, class_id))
            else:
                class_id = np.argmax(class_probabilities)
                confidence = class_probabilities[class_id] * object_probability
                if confidence < threshold:
                    continue
                objects.append(Detection(x - width / 2, y - height / 2, x + width / 2, y + height / 2,
                                         confidence.item(), class_id.item()))
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

    @staticmethod
    def _resize_detections(detections, original_shape, resized_shape=None):
        w, h = original_shape

        if resized_shape:
            w = original_shape[0] / resized_shape[0]
            h = original_shape[1] / resized_shape[1]

        for detection in detections:
            detection.xmin *= w
            detection.xmax *= w
            detection.ymin *= h
            detection.ymax *= h
        return detections

    @staticmethod
    def _resize_detections_letterbox(detections, original_shape, resized_shape):
        scales = [x / y for x, y in zip(resized_shape, original_shape)]
        scale = min(scales)
        scales = (scale / scales[0], scale / scales[1])
        offset = [0.5 * (1 - x) for x in scales]
        for detection in detections:
            detection.xmin = ((detection.xmin - offset[0]) / scales[0]) * original_shape[0]
            detection.xmax = ((detection.xmax - offset[0]) / scales[0]) * original_shape[0]
            detection.ymin = ((detection.ymin - offset[1]) / scales[1]) * original_shape[1]
            detection.ymax = ((detection.ymax - offset[1]) / scales[1]) * original_shape[1]
        return detections

    def postprocess(self, outputs, meta):
        detections = []

        for layer_name in self.yolo_layer_params.keys():
            out_blob = outputs[layer_name]
            layer_params = self.yolo_layer_params[layer_name]
            out_blob.shape = layer_params[0]
            detections += self._parse_yolo_region(out_blob, meta['resized_shape'], layer_params[1], self.threshold)

        detections = self._filter(detections, self.iou_threshold)

        if self.keep_aspect_ratio:
            detections = self._resize_detections_letterbox(detections, meta['original_shape'][1::-1],
                                                           meta['resized_shape'][1::-1])
        else:
            detections = self._resize_detections(detections, meta['original_shape'][1::-1])

        return clip_detections(detections, meta['original_shape'])

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

    def __init__(self, ie, model_path, labels=None, keep_aspect_ratio=False, threshold=0.5, iou_threshold=0.5,
                 anchors=None, masks=None):
        self.anchors = anchors
        self.masks = masks
        super().__init__(ie, model_path, labels, keep_aspect_ratio, threshold, iou_threshold)

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
            class_probabilities = sigmoid(prediction[:, params.coords + 1:].flatten())
            class_probabilities *= np.repeat(object_probabilities, params.classes)

            # filter out the proposals with low confidence score
            keep_idxs = np.nonzero(class_probabilities > threshold)[0]
            class_probabilities = class_probabilities[keep_idxs]
            obj_indx = keep_idxs // params.classes
            class_idx = keep_idxs % params.classes

            for ind, obj_ind in enumerate(obj_indx):
                bbox = prediction[:, :params.coords][obj_ind]
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


class YOLOF(YOLO):
    class Params:
        def __init__(self, classes, num, sides, anchors):
            self.num = num
            self.coords = 4
            self.classes = classes
            self.sides = sides
            self.anchors = anchors

    def _get_output_info(self):
        anchors = ANCHORS['YOLOF']

        output_info = {}
        num = 6
        for i, (name, layer) in enumerate(self.net.outputs.items()):
            shape = layer.shape
            classes = shape[1] // num - 4
            yolo_params = self.Params(classes, num, shape[2:4], anchors)
            output_info[name] = (shape, yolo_params)
        return output_info

    @staticmethod
    def _parse_yolo_region(predictions, input_size, params, threshold):
        def sigmoid(x):
            return 1. / (1. + np.exp(-x))

        objects = []
        bbox_size = params.coords + params.classes
        predictions = permute_to_N_HWA_K(predictions, bbox_size)
        # ------------------------------------------- Parsing YOLO Region output ---------------------------------------
        for prediction in predictions:
            prob = prediction[:, params.coords:].flatten()
            class_probabilities = sigmoid(prob)

            # filter out the proposals with low confidence score
            keep_idxs = np.nonzero(class_probabilities > threshold)[0]
            class_probabilities = class_probabilities[keep_idxs]
            obj_indx = keep_idxs // params.classes
            class_idx = keep_idxs % params.classes

            for ind, obj_ind in enumerate(obj_indx):
                bbox = prediction[:, :params.coords][obj_ind]
                x, y, width, height = bbox[:4]

                row = obj_ind // (params.sides[0] * params.num)
                col = (obj_ind - row * params.sides[0] * params.num) // params.num
                n = (obj_ind - row * params.sides[0] * params.num) % params.num

                # Get absolute coords of center
                anchor_x = params.anchors[2 * n] / input_size[1]
                anchor_y = params.anchors[2 * n + 1] / input_size[0]
                x = x * anchor_x + col / params.sides[1]
                y = y * anchor_y + row / params.sides[0]

                # Value for exp is very big number in some cases so following construction is using here
                try:
                    width = np.exp(width)
                    height = np.exp(height)
                except OverflowError:
                    continue
                width *= anchor_x
                height *= anchor_y

                # Define class_label and cofidence
                label = class_idx[ind]
                confidence = class_probabilities[ind]
                objects.append(Detection(x - width / 2, y - height / 2, x + width / 2, y + height / 2,
                                         confidence.item(), label.item()))

        return objects
