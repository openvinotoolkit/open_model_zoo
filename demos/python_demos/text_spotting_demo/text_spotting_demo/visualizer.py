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

from __future__ import print_function

import cv2
import numpy as np


class Visualizer(object):
    instance_color_palette = np.array([[0, 113, 188],
                                       [216, 82, 24],
                                       [236, 176, 31],
                                       [125, 46, 141],
                                       [118, 171, 47],
                                       [76, 189, 237],
                                       [161, 19, 46],
                                       [76, 76, 76],
                                       [153, 153, 153],
                                       [255, 0, 0],
                                       [255, 127, 0],
                                       [190, 190, 0],
                                       [0, 255, 0],
                                       [0, 0, 255],
                                       [170, 0, 255],
                                       [84, 84, 0],
                                       [84, 170, 0],
                                       [84, 255, 0],
                                       [170, 84, 0],
                                       [170, 170, 0],
                                       [170, 255, 0],
                                       [255, 84, 0],
                                       [255, 170, 0],
                                       [255, 255, 0],
                                       [0, 84, 127],
                                       [0, 170, 127],
                                       [0, 255, 127],
                                       [84, 0, 127],
                                       [84, 84, 127],
                                       [84, 170, 127],
                                       [84, 255, 127],
                                       [170, 0, 127],
                                       [170, 84, 127],
                                       [170, 170, 127],
                                       [170, 255, 127],
                                       [255, 0, 127],
                                       [255, 84, 127],
                                       [255, 170, 127],
                                       [255, 255, 127],
                                       [0, 84, 255],
                                       [0, 170, 255],
                                       [0, 255, 255],
                                       [84, 0, 255],
                                       [84, 84, 255],
                                       [84, 170, 255],
                                       [84, 255, 255],
                                       [170, 0, 255],
                                       [170, 84, 255],
                                       [170, 170, 255],
                                       [170, 255, 255],
                                       [255, 0, 255],
                                       [255, 84, 255],
                                       [255, 170, 255],
                                       [42, 0, 0],
                                       [84, 0, 0],
                                       [127, 0, 0],
                                       [170, 0, 0],
                                       [212, 0, 0],
                                       [255, 0, 0],
                                       [0, 42, 0],
                                       [0, 84, 0],
                                       [0, 127, 0],
                                       [0, 170, 0],
                                       [0, 212, 0],
                                       [0, 255, 0],
                                       [0, 0, 42],
                                       [0, 0, 84],
                                       [0, 0, 127],
                                       [0, 0, 170],
                                       [0, 0, 212],
                                       [0, 0, 255],
                                       [0, 0, 0],
                                       [36, 36, 36],
                                       [72, 72, 72],
                                       [109, 109, 109],
                                       [145, 145, 145],
                                       [182, 182, 182],
                                       [218, 218, 218],
                                       [255, 255, 255]], dtype=np.uint8)

    class_color_palette = np.asarray([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

    def __init__(self, class_labels, show_boxes=False,
                 show_masks=True, show_scores=False):
        super().__init__()
        self.class_labels = class_labels
        self.show_masks = show_masks
        self.show_boxes = show_boxes
        self.show_scores = show_scores

    def __call__(self, image, boxes, classes, scores, segms, texts, ids=None):
        if self.show_masks and segms is not None:
            image = self.overlay_masks(image, segms, ids)

        if self.show_boxes:
            image = self.overlay_boxes(image, boxes, classes)

        image = self.overlay_class_names(image, boxes, texts, scores, show_score=self.show_scores)

        return image

    def compute_colors_for_labels(self, labels):
        colors = labels[:, None] * self.class_color_palette
        colors = (colors % 255).astype(np.uint8)
        return colors

    def overlay_boxes(self, image, boxes, classes):
        colors = self.compute_colors_for_labels(classes).tolist()
        for box, color in zip(boxes, colors):
            box = box.astype(int)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 1
            )
        return image

    def overlay_masks(self, image, masks, ids=None):
        segments_image = image.copy()
        aggregated_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        aggregated_colored_mask = np.zeros(image.shape, dtype=np.uint8)
        black = np.zeros(3, dtype=np.uint8)

        all_contours = []
        for i, mask in enumerate(masks):
            color_idx = i if ids is None else ids[i]
            mask_color = self.instance_color_palette[color_idx % len(self.instance_color_palette)].tolist()

            contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
            if contours:
                all_contours.append(contours[0])

            cv2.bitwise_or(aggregated_mask, mask, dst=aggregated_mask)
            cv2.bitwise_or(aggregated_colored_mask, np.asarray(mask_color, dtype=np.uint8),
                           dst=aggregated_colored_mask, mask=mask)

        # Fill the area occupied by all instances with a colored instances mask image.
        cv2.bitwise_and(segments_image, black, dst=segments_image, mask=aggregated_mask)
        cv2.bitwise_or(segments_image, aggregated_colored_mask, dst=segments_image, mask=aggregated_mask)
        # Blend original image with the one, where instances are colored.
        # As a result instances masks become transparent.
        cv2.addWeighted(image, 0.5, segments_image, 0.5, 0, dst=image)

        cv2.drawContours(image, all_contours, -1, (0, 0, 0))

        return image

    def overlay_class_names(self, image, boxes, texts, scores, show_score=True):
        template = '{}: {:.2f}' if show_score else '{}'
        white = (255, 255, 255)

        for box, score, label in zip(boxes, scores, texts):
            s = template.format(label, score)
            textsize = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            position = ((box[:2] + box[2:] - textsize) / 2).astype(int)
            cv2.putText(image, s, tuple(position), cv2.FONT_HERSHEY_SIMPLEX, .5, white, 1)

        return image
