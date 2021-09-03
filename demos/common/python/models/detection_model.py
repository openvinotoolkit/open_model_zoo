"""
 Copyright (c) 2021 Intel Corporation

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
from .image_model import ImageModel
from .utils import load_labels, clip_detections


class DetectionModel(ImageModel):
    def __init__(self, ie, model_path, input_transform=None, resize_type='default',
                 labels=None, threshold=None, iou_threshold=None):
        super().__init__(ie, model_path, input_transform=input_transform, resize_type=resize_type)
        if isinstance(labels, (list, tuple)):
            self.labels = labels
        else:
            self.labels = load_labels(labels) if labels else None

        self.threshold = threshold
        self.iou_threshold = iou_threshold

    def _resize_detections(self, detections, meta):
        resized_shape = meta['resized_shape']
        original_shape = meta['original_shape']

        if self.resize_type=='letterbox':
            detections = resize_detections_letterbox(detections, original_shape[1::-1], resized_shape[1::-1])
        elif self.resize_type == 'keep_aspect_ratio':
            detections = resize_detections_with_aspect_ratio(detections, original_shape[1::-1], resized_shape[1::-1], (self.w, self.h))
        elif self.resize_type == 'default':
            detections = resize_detections(detections, original_shape[1::-1])
        else:
            raise RuntimeError('Unknown resize type {}'.format(self.resize_type))
        return clip_detections(detections, original_shape)


def resize_detections(detections, original_image_size):
    for detection in detections:
        detection.xmin *= original_image_size[0]
        detection.xmax *= original_image_size[0]
        detection.ymin *= original_image_size[1]
        detection.ymax *= original_image_size[1]
    return detections

def resize_detections_with_aspect_ratio(detections, original_image_size, resized_image_size, model_input_size):
    scale_x = model_input_size[0] / resized_image_size[0] * original_image_size[0]
    scale_y = model_input_size[1] / resized_image_size[1] * original_image_size[1]
    for detection in detections:
        detection.xmin *= scale_x
        detection.xmax *= scale_x
        detection.ymin *= scale_y
        detection.ymax *= scale_y
    return detections

def resize_detections_letterbox(detections, original_image_size, resized_image_size):
    scales = [x / y for x, y in zip(resized_image_size, original_image_size)]
    scale = min(scales)
    scales = (scale / scales[0], scale / scales[1])
    offset = [0.5 * (1 - x) for x in scales]
    for detection in detections:
        detection.xmin = ((detection.xmin - offset[0]) / scales[0]) * original_image_size[0]
        detection.xmax = ((detection.xmax - offset[0]) / scales[0]) * original_image_size[0]
        detection.ymin = ((detection.ymin - offset[1]) / scales[1]) * original_image_size[1]
        detection.ymax = ((detection.ymax - offset[1]) / scales[1]) * original_image_size[1]
    return detections
