"""
 Copyright (C) 2021 Intel Corporation

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

from .model import Model
from .utils import resize_image, DetectionWithLandmarks


class ProposalModel(Model):
    def __init__(self, ie, model_path, score_treshold=0.6, nms_threshold=0.5, min_face_size=10):
        super().__init__(ie, model_path)
        self.score_threshold = score_treshold
        self.nms_threshold = nms_threshold
        self.min_face_size = min_face_size
        self.image_blob_name = next(iter(self.net.input_info))
        for name, blob in self.net.outputs.items():
            if blob.shape[1] == 2:
                self.prob_blob_name = name
            elif blob.shape[1] == 4:
                self.bbox_blob_name = name

    def preprocess(self, inputs):
        # MTCNN input layout is NCWH
        image = inputs
        n, c, w, h = self.net.input_info[self.image_blob_name].input_data.shape
        resized_image = resize_image(image, (w, h))
        resized_image = resized_image.transpose((2, 1, 0))
        resized_image = resized_image.reshape((n, c, w, h))
        dict_inputs = {self.image_blob_name: resized_image}
        meta = {'original_shape': image.shape,
                'resized_shape': resized_image.shape,
                'resize_scale': 1/self.scale}
        return dict_inputs, meta

    def postprocess(self, outputs, meta):
        bboxes = outputs[self.bbox_blob_name]
        scores = outputs[self.prob_blob_name]
        height, width, _ = meta['original_shape']
        resize_scale = meta['resize_scale']

        detections = self._generate_detections(bboxes[0], scores[0], width, height, resize_scale, self.score_threshold)
        detections = nms(detections, self.nms_threshold)
        return detections, meta

    def postprocess_all(self, detections):
        detections = nms(detections, 0.7)
        return detections

    @staticmethod
    def _generate_detections(bboxes, scores, width, height, scale, score_threshold):
        def square_box(boxes):
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            a = np.maximum(w, h)
            shift = 0.5 * (np.array([w, h]) - np.repeat([a], 2, axis=0)).T
            boxes[:, :2] += shift
            boxes[:, 2:4] -= shift
            return boxes

        out_side = max(bboxes.shape[1:])
        in_side = 2 * out_side + 11
        stride = 2
        if out_side != 1:
            stride = float(in_side - 12) / (out_side - 1)
        (x, y) = np.where(scores[1] >= score_threshold)
        if x.size == 0:
            return []

        kernels = np.array([x, y]).T
        kernels = np.concatenate((np.fix((stride * kernels) * scale),
                                  np.fix((stride * kernels + 11) * scale)),
                                 axis=1)
        offset = bboxes[:, x, y].T
        score = np.array([scores[1][x, y]]).T
        boxes = kernels + offset * 12.0 * scale
        boxes = np.concatenate((boxes, score), axis=1)
        boxes = square_box(boxes)
        result = []
        for box in boxes:
            box[0] = max(0, box[0])
            box[1] = max(0, box[1])
            box[2] = min(width, box[2])
            box[3] = min(height, box[3])
            if box[2] > box[0] and box[3] > box[1]:
                box[0] = max(0, box[0])
                box[1] = max(0, box[1])
                box[2] = min(width, box[2])
                box[3] = min(height, box[3])
                result.append(box)

        return result

    def calc_scales(self, image):
        pr_scale = 12.0 / self.min_face_size
        h, w = image.shape[:2]
        scales = []
        factor = 0.709
        factor_count = 0
        minl = min(h, w)
        while minl >= 12:
            scales.append(pr_scale*pow(factor, factor_count))
            minl *= factor
            factor_count += 1
        return scales


class RefineModel(Model):
    def __init__(self, ie, model_path, score_threshold=0.7, nms_threshold=0.7):
        super().__init__(ie, model_path)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.image_blob_name = next(iter(self.net.input_info))
        for name, blob in self.net.outputs.items():
            if blob.shape[1] == 2:
                self.prob_blob_name = name
            elif blob.shape[1] == 4:
                self.bbox_blob_name = name
        self.n, self.c, self.w, self.h = self.net.input_info[self.image_blob_name].input_data.shape

    def preprocess(self, inputs):
        image = inputs['image']
        crops = np.array(inputs['crop'])
        if len(crops.shape) == 1:
            crops = np.expand_dims(crops, axis=0)
        meta = {'crop': crops,
                'original_shape': image.shape}
        self.n, self.c, self.w, self.h = self.net.input_info[self.image_blob_name].input_data.shape
        batched_images = []

        for crop in crops:
            x1, y1, x2, y2 = [int(x) for x in crop[:4]]
            crop_image = image[y1:y2, x1:x2]
            resized_image = resize_image(crop_image, (self.w, self.h))
            resized_image = resized_image.transpose((2, 1, 0))
            resized_image = resized_image.reshape((1, self.c, self.w, self.h))
            batched_images.extend(resized_image)

        while len(batched_images) < self.n:
            batched_images.extend(np.ones(shape=(1, self.c, self.w, self.h), dtype=resized_image.dtype))
        dict_inputs = {self.image_blob_name: batched_images}
        return dict_inputs, meta

    def postprocess(self, outputs, meta):
        bboxes = outputs[self.bbox_blob_name]
        scores = outputs[self.prob_blob_name]
        height, width, _ = meta['original_shape']
        crops = meta['crop']
        total_bboxes = crops.shape[0]
        bboxes = bboxes[:total_bboxes, :]
        scores = scores[:total_bboxes, :]
        detections = self._generate_detections(bboxes, scores, width, height, crops, self.score_threshold)
        return detections, meta

    @staticmethod
    def _generate_detections(bboxes, scores, width, height, crop, score_threshold):
        def square_box(boxes):
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            a = np.maximum(w, h)
            shift = 0.5 * (np.array([w, h]) - np.repeat([a], 2, axis=0)).T
            boxes[:, :2] += shift
            boxes[:, 2:4] -= shift
            return boxes

        if len(scores.shape) == 1:
            bboxes = np.expand_dims(bboxes, axis=1)
            scores = np.expand_dims(scores, axis=1)

        (keep,) = np.where(scores[:, 1] >= score_threshold)
        if keep.size == 0:
            return []
        bboxes = bboxes[keep, :]
        scores = scores[keep, 1]
        scores = np.expand_dims(scores, axis=1)
        crop = crop[keep, :]

        w = crop[:, 2] - crop[:, 0]
        h = crop[:, 3] - crop[:, 1]
        bboxes[:, 0] = bboxes[:, 0] * w + crop[:, 0]
        bboxes[:, 1] = bboxes[:, 1] * w + crop[:, 1]
        bboxes[:, 2] = bboxes[:, 2] * h + crop[:, 2]
        bboxes[:, 3] = bboxes[:, 3] * h + crop[:, 3]

        boxes = np.concatenate((bboxes, scores), axis=1)
        boxes = square_box(boxes)

        result = []
        for box in boxes:
            box[0] = max(0, box[0])
            box[1] = max(0, box[1])
            box[2] = min(width, box[2])
            box[3] = min(height, box[3])
            if box[2] > box[0] and box[3] > box[1]:
                result.append(box)

        return result

    def postprocess_all(self, detections):
        detections = nms(detections, self.nms_threshold)
        return detections


class OutputModel(Model):
    def __init__(self, ie, model_path, score_threshold=0.7, nms_threshold=0.7):
        super().__init__(ie, model_path)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.image_blob_name = next(iter(self.net.input_info))
        for name, blob in self.net.outputs.items():
            if blob.shape[1] == 2:
                self.prob_blob_name = name
            elif blob.shape[1] == 4:
                self.bbox_blob_name = name
            elif blob.shape[1] == 10:
                self.landmarks_blob_name = name
        self.n, self.c, self.w, self.h = self.net.input_info[self.image_blob_name].input_data.shape

    def preprocess(self, inputs):
        image = inputs['image']
        crops = np.array(inputs['crop'])
        if len(crops.shape) == 1:
            crops = np.expand_dims(crops, axis=0)
        meta = {'crop': crops,
                'original_shape': image.shape}
        self.n, self.c, self.w, self.h = self.net.input_info[self.image_blob_name].input_data.shape
        batched_images = []
        for crop in crops:
            x1, y1, x2, y2 = [int(x) for x in crop[:4]]
            crop_image = image[y1:y2, x1:x2]
            resized_image = resize_image(crop_image, (self.w, self.h))
            resized_image = resized_image.transpose((2, 1, 0))
            resized_image = resized_image.reshape((1, self.c, self.w, self.h))
            batched_images.extend(resized_image)

        while len(batched_images) < self.n:
            batched_images.extend(np.ones(shape=(1, self.c, self.w, self.h), dtype=resized_image.dtype))

        dict_inputs = {self.image_blob_name: batched_images}
        return dict_inputs, meta

    def postprocess(self, outputs, meta):
        bboxes = outputs[self.bbox_blob_name]
        scores = outputs[self.prob_blob_name]
        landmarks = outputs[self.landmarks_blob_name]
        crops = meta['crop']
        height, width, _ = meta['original_shape']
        total_bboxes = crops.shape[0]
        bboxes = bboxes[:total_bboxes, :]
        scores = scores[:total_bboxes, :]

        detections = self._generate_detections(bboxes, scores, landmarks, width, height, crops, self.score_threshold)
        return detections, meta

    @staticmethod
    def _generate_detections(bboxes, scores, landmarks, width, height, crop, score_threshold):
        if len(scores.shape) == 1:
            bboxes = np.expand_dims(bboxes, axis=1)
            scores = np.expand_dims(scores, axis=1)
            landmarks = np.expand_dims(landmarks, axis=1)

        (keep,) = np.where(scores[:, 1] >= score_threshold)
        if keep.size == 0:
            return []
        bboxes = bboxes[keep, :]
        scores = scores[keep, 1]
        scores = np.expand_dims(scores, axis=1)
        landmarks = landmarks[keep, :]
        crop = crop[keep, :]

        w = crop[:, 2] - crop[:, 0]
        h = crop[:, 3] - crop[:, 1]
        bboxes[:, 0] = bboxes[:, 0] * w + crop[:, 0]
        bboxes[:, 1] = bboxes[:, 1] * w + crop[:, 1]
        bboxes[:, 2] = bboxes[:, 2] * h + crop[:, 2]
        bboxes[:, 3] = bboxes[:, 3] * h + crop[:, 3]
        landmarks[:, :5] = landmarks[:, :5] * np.expand_dims(w, axis=1) + np.expand_dims(crop[:, 0], axis=1)
        landmarks[:, 5:] = landmarks[:, 5:] * np.expand_dims(h, axis=1) + np.expand_dims(crop[:, 1], axis=1)

        classes = np.ones(shape=scores.shape)
        boxes = np.concatenate((bboxes, scores, classes, landmarks), axis=1)

        result = []
        for box in boxes:
            box[0] = max(0, box[0])
            box[1] = max(0, box[1])
            box[2] = min(width, box[2])
            box[3] = min(height, box[3])
            if box[2] > box[0] and box[3] > box[1]:
                result.append(box)

        return result

    def postprocess_all(self, detections):
        detections = nms(detections, self.nms_threshold, False)
        return detections

    @staticmethod
    def make_detections(detections):
        return [DetectionWithLandmarks(*detection[:6], detection[6:11], detection[11:]) for detection in detections]


def nms(bboxes, threshold, iou=True):
    if len(bboxes) == 0:
        return bboxes
    boxes = np.array(bboxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]
    area = np.multiply(x2-x1+1, y2-y1+1)
    indexes = np.array(scores.argsort())
    keep = []
    while len(indexes) > 0:
        xx1 = np.maximum(x1[indexes[-1]], x1[indexes[0:-1]])
        yy1 = np.maximum(y1[indexes[-1]], y1[indexes[0:-1]])
        xx2 = np.minimum(x2[indexes[-1]], x2[indexes[0:-1]])
        yy2 = np.minimum(y2[indexes[-1]], y2[indexes[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if iou:
            o = inter / (area[indexes[-1]] + area[indexes[0:-1]] - inter)
        else:
            o = inter / np.minimum(area[indexes[-1]], area[indexes[0:-1]])

        keep.append(indexes[-1])
        indexes = indexes[np.where(o <= threshold)[0]]
    return boxes[keep].tolist()
