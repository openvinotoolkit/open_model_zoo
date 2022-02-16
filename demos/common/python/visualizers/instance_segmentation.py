"""
 Copyright (c) 2022 Intel Corporation

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

from .drawing_utils import ColorPalette


class InstanceSegmentationVisualizer:
    def __init__(self, labels=None, show_boxes=False, show_scores=False):
        colors_num = len(labels) if labels else 80
        self.labels = labels
        self.palette = ColorPalette(colors_num)
        self.show_boxes = show_boxes
        self.show_scores = show_scores

    def __call__(self, image, boxes, classes, scores, masks=None, ids=None, texts=None):
        result = image.copy()

        if masks is not None:
            result = self.overlay_masks(result, masks, ids)
        if self.show_boxes:
            result = self.overlay_boxes(result, boxes, classes)

        result = self.overlay_labels(result, boxes, classes, scores, texts)
        return result

    def overlay_masks(self, image, masks, ids=None):
        segments_image = image.copy()
        aggregated_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        aggregated_colored_mask = np.zeros(image.shape, dtype=np.uint8)
        all_contours = []

        for i, mask in enumerate(masks):
            contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
            if contours:
                all_contours.append(contours[0])

            mask_color = self.palette[i if ids is None else ids[i]]
            cv2.bitwise_or(aggregated_mask, mask, dst=aggregated_mask)
            cv2.bitwise_or(aggregated_colored_mask, mask_color, dst=aggregated_colored_mask, mask=mask)

        # Fill the area occupied by all instances with a colored instances mask image
        cv2.bitwise_and(segments_image, (0, 0, 0), dst=segments_image, mask=aggregated_mask)
        cv2.bitwise_or(segments_image, aggregated_colored_mask, dst=segments_image, mask=aggregated_mask)

        cv2.addWeighted(image, 0.5, segments_image, 0.5, 0, dst=image)
        cv2.drawContours(image, all_contours, -1, (0, 0, 0))
        return image

    def overlay_boxes(self, image, boxes, classes):
        for box, class_id in zip(boxes, classes):
            color = self.palette[class_id]
            box = box.astype(int)
            top_left, bottom_right = box[:2], box[2:]
            image = cv2.rectangle(image, top_left, bottom_right, color, 2)
        return image

    def overlay_labels(self, image, boxes, classes, scores, texts=None):
        if texts:
            labels = texts
        elif self.labels:
            labels = (self.labels[class_id] for class_id in classes)
        else:
            raise RuntimeError('InstanceSegmentationVisualizer must contain either labels or texts to display')
        template = '{}: {:.2f}' if self.show_scores else '{}'

        for box, score, label in zip(boxes, scores, labels):
            text = template.format(label, score)
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            position = ((box[:2] + box[2:] - textsize) / 2).astype(np.int32)
            cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return image
