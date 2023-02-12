#!/usr/bin/env python3
"""
 Copyright (C) 2018-2022 Intel Corporation

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

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/model_tools/src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'demos/common/python'))

from openvino.model_zoo.model_api.models import SegmentationModel


class SegmentationVisualizer:
    def __init__(self):
        pascal_palette_path = Path(__file__).resolve().parents[1] /\
            'data/palettes/pascal_voc_21cl_colors.txt'
        self.color_palette = self.get_palette_from_file(pascal_palette_path)
        self.color_map = self.create_color_map()

    def get_palette_from_file(self, colors_path):
        with open(colors_path, 'r') as file:
            colors = []
            for line in file.readlines():
                values = line[line.index('(')+1:line.index(')')].split(',')
                colors.append([int(v.strip()) for v in values])
            return colors

    def create_color_map(self):
        classes = np.array(self.color_palette, dtype=np.uint8)[:, ::-1] # RGB to BGR
        color_map = np.zeros((256, 1, 3), dtype=np.uint8)
        classes_num = len(classes)
        color_map[:classes_num, 0, :] = classes
        color_map[classes_num:, 0, :] = np.random.uniform(0, 255, size=(256-classes_num, 3))
        return color_map

    def apply_color_map(self, input):
        input_3d = cv2.merge([input, input, input])
        return cv2.LUT(input_3d, self.color_map)



def render_segmentation(frame, masks, visualiser):
    output = visualiser.apply_color_map(masks)
    return cv2.addWeighted(frame, 0.5, output, 0.5, 0)


def main():
    if len(sys.argv) != 2:
        raise RuntimeError(f'Usage: {sys.argv[0]} <path_to_image>')
    segmentor = SegmentationModel.create_model('fastseg-small')
    image = cv2.imread(sys.argv[1])
    if image is None:
        raise RuntimeError('Failed to read the image')
    mask = segmentor(image)
    masked = render_segmentation(image, mask, SegmentationVisualizer())
    cv2.imshow('Detection Results', masked)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
