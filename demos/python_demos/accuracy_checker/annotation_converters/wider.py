"""
 Copyright (c) 2018 Intel Corporation

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
import pathlib

from accuracy_checker.representation import DetectionAnnotation
from accuracy_checker.utils import check_exists

from .format_converter import BaseFormatConverter


class WiderFormatConverter(BaseFormatConverter):
    __provider__ = 'wider'

    def convert(self, annotation_file: str, label_start=1):
        annotation_file = pathlib.Path(annotation_file).absolute()
        check_exists(annotation_file.as_posix())
        annotations = []

        with open(annotation_file.as_posix(), 'r') as wider_annotation:
            image_annotations = wider_annotation.read()
            image_annotations = image_annotations.split('\n')
            image_ids = []
            for image_id, line in enumerate(image_annotations):
                if '.jpg' in line:
                    image_ids.append(image_id)

            for image_id in image_ids:
                identifier = image_annotations[image_id]
                bbox_count = image_annotations[image_id+1]
                bbox_lines = image_annotations[image_id+2:image_id+2+int(bbox_count)]
                x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
                for bbox in bbox_lines:
                    x_min, y_min, width, height = bbox.split(' ')[0:4]
                    x_mins.append(int(x_min))
                    y_mins.append(int(y_min))
                    x_maxs.append(int(x_min) + int(width))
                    y_maxs.append(int(y_min) + int(height))
                annotations.append(DetectionAnnotation(identifier, [int(label_start)]*len(x_mins),
                                                       x_mins, y_mins, x_maxs, y_maxs))
        return annotations, {'label_map': {0: '__background__', int(label_start): 'face'}, 'background_label': 0}
