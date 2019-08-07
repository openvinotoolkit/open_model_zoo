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

from ..config import NumberField, PathField
from ..representation import DetectionAnnotation
from ..utils import convert_bboxes_xywh_to_x1y1x2y2, read_txt, check_file_existence

from .format_converter import BaseFormatConverter, ConverterReturn


class WiderFormatConverter(BaseFormatConverter):
    __provider__ = 'wider'
    annotation_types = (DetectionAnnotation, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'annotation_file': PathField(
                description="Path to xml file, which contains ground truth data in WiderFace dataset format."
            ),
            'label_start': NumberField(
                value_type=int, optional=True, default=1,
                description="Specifies face label index in label map. Default value is 1. "
                            "You can provide another value, if you want to use this"
            ),
            'images_dir': PathField(
                is_directory=True, optional=True,
                description='path to dataset images, used only for content existence check'
            )
        })

        return parameters

    def configure(self):
        self.annotation_file = self.get_value_from_config('annotation_file')
        self.label_start = self.get_value_from_config('label_start')
        self.images_dir = self.get_value_from_config('images_dir') or self.annotation_file.parent

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        image_annotations = read_txt(self.annotation_file)
        content_errors = None if not check_content else []
        image_ids = []
        for image_id, line in enumerate(image_annotations):
            if '.jpg' in line:
                image_ids.append(image_id)

        annotations = []
        num_iterations = len(image_ids)
        for index, image_id in enumerate(image_ids):
            identifier = image_annotations[image_id]
            if check_content:
                if not check_file_existence(self.images_dir / identifier):
                    content_errors.append('{}: does not exist'.format(self.images_dir / identifier))

            bbox_count = image_annotations[image_id + 1]
            bbox_lines = image_annotations[image_id + 2:image_id + 2 + int(bbox_count)]

            x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
            for bbox in bbox_lines:
                x_min, y_min, x_max, y_max = convert_bboxes_xywh_to_x1y1x2y2(*(map(float, (bbox.split(' ')[0:4]))))
                x_mins.append(x_min)
                y_mins.append(y_min)
                x_maxs.append(x_max)
                y_maxs.append(y_max)

            annotations.append(DetectionAnnotation(
                identifier, [self.label_start] * len(x_mins),
                x_mins, y_mins, x_maxs, y_maxs
            ))
            if progress_callback and index % progress_interval == 0:
                progress_callback(index * 100 / num_iterations)

        meta = {'label_map': {0: '__background__', self.label_start: 'face'}, 'background_label': 0}

        return ConverterReturn(annotations, meta, content_errors)
