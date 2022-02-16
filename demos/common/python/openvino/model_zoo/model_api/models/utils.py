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

import cv2
import numpy as np
import math


class Detection:
    def __init__(self, xmin, ymin, xmax, ymax, score, id):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.score = score
        self.id = id

    def bottom_left_point(self):
        return self.xmin, self.ymin

    def top_right_point(self):
        return self.xmax, self.ymax

    def get_coords(self):
        return self.xmin, self.ymin, self.xmax, self.ymax


def clip_detections(detections, size):
    for detection in detections:
        detection.xmin = max(int(detection.xmin), 0)
        detection.ymin = max(int(detection.ymin), 0)
        detection.xmax = min(int(detection.xmax), size[1])
        detection.ymax = min(int(detection.ymax), size[0])
    return detections


class DetectionWithLandmarks(Detection):
    def __init__(self, xmin, ymin, xmax, ymax, score, id, landmarks_x, landmarks_y):
        super().__init__(xmin, ymin, xmax, ymax, score, id)
        self.landmarks = []
        for x, y in zip(landmarks_x, landmarks_y):
            self.landmarks.append((x, y))


class OutputTransform:
    def __init__(self, input_size, output_resolution):
        self.output_resolution = output_resolution
        if self.output_resolution:
            self.new_resolution = self.compute_resolution(input_size)

    def compute_resolution(self, input_size):
        self.input_size = input_size
        size = self.input_size[::-1]
        self.scale_factor = min(self.output_resolution[0] / size[0],
                                self.output_resolution[1] / size[1])
        return self.scale(size)

    def resize(self, image):
        if not self.output_resolution:
            return image
        curr_size = image.shape[:2]
        if curr_size != self.input_size:
            self.new_resolution = self.compute_resolution(curr_size)
        if self.scale_factor == 1:
            return image
        return cv2.resize(image, self.new_resolution)

    def scale(self, inputs):
        if not self.output_resolution or self.scale_factor == 1:
            return inputs
        return (np.array(inputs) * self.scale_factor).astype(np.int32)


class InputTransform:
    def __init__(self, reverse_input_channels=False, mean_values=None, scale_values=None):
        self.reverse_input_channels = reverse_input_channels
        self.is_trivial = not (reverse_input_channels or mean_values or scale_values)
        self.means = np.array(mean_values, dtype=np.float32) if mean_values else np.array([0., 0., 0.])
        self.std_scales = np.array(scale_values, dtype=np.float32) if scale_values else np.array([1., 1., 1.])

    def __call__(self, inputs):
        if self.is_trivial:
            return inputs
        if self.reverse_input_channels:
            inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB)
        return (inputs - self.means) / self.std_scales


def load_labels(label_file):
    with open(label_file, 'r') as f:
        labels_map = [x.strip() for x in f]
    return labels_map


def resize_image(image, size, keep_aspect_ratio=False, interpolation=cv2.INTER_LINEAR):
    if not keep_aspect_ratio:
        resized_frame = cv2.resize(image, size, interpolation=interpolation)
    else:
        h, w = image.shape[:2]
        scale = min(size[1] / h, size[0] / w)
        resized_frame = cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)
    return resized_frame


def resize_image_with_aspect(image, size, interpolation=cv2.INTER_LINEAR):
    return resize_image(image, size, keep_aspect_ratio=True, interpolation=interpolation)


def pad_image(image, size):
    h, w = image.shape[:2]
    if h != size[1] or w != size[0]:
        image = np.pad(image, ((0, size[1] - h), (0, size[0] - w), (0, 0)),
                               mode='constant', constant_values=0)
    return image


def resize_image_letterbox(image, size, interpolation=cv2.INTER_LINEAR):
    ih, iw = image.shape[0:2]
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = cv2.resize(image, (nw, nh), interpolation=interpolation)
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    resized_image = np.pad(image, ((dy, dy + (h - nh) % 2), (dx, dx + (w - nw) % 2), (0, 0)),
                           mode='constant', constant_values=128)
    return resized_image


def crop_resize(image, size):
    desired_aspect_ratio = size[1] / size[0] # width / height
    if desired_aspect_ratio == 1:
        if (image.shape[0] > image.shape[1]):
            offset = (image.shape[0] - image.shape[1]) // 2
            cropped_frame = image[offset:image.shape[1] + offset]
        else:
            offset = (image.shape[1] - image.shape[0]) // 2
            cropped_frame = image[:, offset:image.shape[0] + offset]
    elif desired_aspect_ratio < 1:
        new_width = math.floor(image.shape[0] * desired_aspect_ratio)
        offset = (image.shape[1] - new_width) // 2
        cropped_frame = image[:, offset:new_width + offset]
    elif desired_aspect_ratio > 1:
        new_height = math.floor(image.shape[1] / desired_aspect_ratio)
        offset = (image.shape[0] - new_height) // 2
        cropped_frame = image[offset:new_height + offset]

    return cv2.resize(cropped_frame, size)


RESIZE_TYPES = {
    'crop' : crop_resize,
    'standard': resize_image,
    'fit_to_window': resize_image_with_aspect,
    'fit_to_window_letterbox': resize_image_letterbox,
}


INTERPOLATION_TYPES = {
    'LINEAR': cv2.INTER_LINEAR,
    'CUBIC': cv2.INTER_CUBIC,
    'NEAREST': cv2.INTER_NEAREST,
    'AREA': cv2.INTER_AREA,
}


def nms(x1, y1, x2, y2, scores, thresh, include_boundaries=False, keep_top_k=None):
    b = 1 if include_boundaries else 0
    areas = (x2 - x1 + b) * (y2 - y1 + b)
    order = scores.argsort()[::-1]

    if keep_top_k:
        order = order[:keep_top_k]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + b)
        h = np.maximum(0.0, yy2 - yy1 + b)
        intersection = w * h

        union = (areas[i] + areas[order[1:]] - intersection)
        overlap = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)

        order = order[np.where(overlap <= thresh)[0] + 1]

    return keep


def softmax(logits, axis=None):
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=axis)
