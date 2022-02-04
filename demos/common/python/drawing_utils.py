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

import random
import colorsys

import cv2
import numpy as np


class ColorPalette:
    def __init__(self, n, rng=None):
        if n == 0:
            raise ValueError('ColorPalette accepts only the positive number of colors')
        if rng is None:
            rng = random.Random(0xACE)  # nosec - disable B311:random check

        candidates_num = 100
        hsv_colors = [(1.0, 1.0, 1.0)]
        for _ in range(1, n):
            colors_candidates = [(rng.random(), rng.uniform(0.8, 1.0), rng.uniform(0.5, 1.0))
                                 for _ in range(candidates_num)]
            min_distances = [self.min_distance(hsv_colors, c) for c in colors_candidates]
            arg_max = np.argmax(min_distances)
            hsv_colors.append(colors_candidates[arg_max])

        self.palette = [self.hsv2rgb(*hsv) for hsv in hsv_colors]

    @staticmethod
    def dist(c1, c2):
        dh = min(abs(c1[0] - c2[0]), 1 - abs(c1[0] - c2[0])) * 2
        ds = abs(c1[1] - c2[1])
        dv = abs(c1[2] - c2[2])
        return dh * dh + ds * ds + dv * dv

    @classmethod
    def min_distance(cls, colors_set, color_candidate):
        distances = [cls.dist(o, color_candidate) for o in colors_set]
        return np.min(distances)

    @staticmethod
    def hsv2rgb(h, s, v):
        return tuple(round(c * 255) for c in colorsys.hsv_to_rgb(h, s, v))

    def __getitem__(self, n):
        return self.palette[n % len(self.palette)]

    def __len__(self):
        return len(self.palette)


class InstanceSegmentationVisualizer:
    def __init__(self, labels, show_boxes=False, show_scores=False):
        self.palette = ColorPalette(len(labels))
        self.labels = labels
        self.show_boxes = show_boxes
        self.show_scores = show_scores

    def __call__(self, image, boxes, classes, scores, masks=None, ids=None):
        result = image.copy()

        if masks is not None:
            result = self.overlay_masks(result, masks, ids)
        if self.show_boxes:
            result = self.overlay_boxes(result, boxes, classes)

        result = self.overlay_labels(result, boxes, classes, scores)
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

    def overlay_labels(self, image, boxes, classes, scores):
        labels = (self.labels[class_id] for class_id in classes)
        template = '{}: {:.2f}' if self.show_scores else '{}'

        for box, score, label in zip(boxes, scores, labels):
            text = template.format(label, score)
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            position = ((box[:2] + box[2:] - textsize) / 2).astype(np.int32)
            cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return image
