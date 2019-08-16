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

import numpy as np
from ..representation import TextDetectionAnnotation, CharacterRecognitionAnnotation
from ..utils import read_txt, check_file_existence
from .format_converter import FileBasedAnnotationConverter, DirectoryBasedAnnotationConverter, ConverterReturn
from ..config import PathField


def box_to_points(box):
    return np.array([[box[0][0], box[0][1]], [box[1][0], box[0][1]], [box[1][0], box[1][1]], [box[0][0], box[1][1]]])


class ICDAR15DetectionDatasetConverter(DirectoryBasedAnnotationConverter):
    __provider__ = 'icdar_detection'
    annotation_types = (TextDetectionAnnotation, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update(
            {
                'images_dir': PathField(
                    is_directory=True, optional=True,
                    description='path to dataset images, used only for content existence check'
                )
            }
        )
        return parameters

    def configure(self):
        super().configure()
        self.images_dir = self.get_value_from_config('images_dir')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotations = []
        content_errors = None if not check_content else []
        self.images_dir = self.images_dir or self.data_dir
        files = list(self.data_dir.iterdir())
        num_iterations = len(files)

        for gt_id, gt_file in enumerate(files):
            gt_file_name = str(gt_file.parts[-1])
            identifier = '{}.jpg'.format(gt_file_name.split('gt_')[-1].split('.txt')[0])
            if check_content:
                if not check_file_existence(self.images_dir / identifier):
                    content_errors.append('{}: does not exist'.format(self.images_dir / identifier))

            all_points, transcriptions, difficult = [], [], []

            for text_area in read_txt(gt_file):
                text_annotation = text_area.split(',')
                transcription = text_annotation[-1]
                num_coords = 8 if len(text_annotation) >= 8 else 4
                coords = text_annotation[:num_coords]
                points = np.reshape(list(map(float, coords)), (-1, 2))
                if num_coords == 4:
                    points = box_to_points(points)
                if transcription == '###':
                    difficult.append(len(transcriptions))
                all_points.append(points)
                transcriptions.append(transcription)
            annotation = TextDetectionAnnotation(identifier, all_points, transcriptions)
            annotation.metadata['difficult_boxes'] = difficult
            annotations.append(annotation)

            if progress_callback is not None and gt_id % progress_interval == 0:
                progress_callback(gt_id / num_iterations * 100)

        return ConverterReturn(annotations, None, content_errors)


class ICDAR13RecognitionDatasetConverter(FileBasedAnnotationConverter):
    __provider__ = 'icdar13_recognition'
    annotation_types = (CharacterRecognitionAnnotation, )

    supported_symbols = '0123456789abcdefghijklmnopqrstuvwxyz'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update(
            {
                'images_dir': PathField(
                    is_directory=True, optional=True,
                    description='path to dataset images, used only for content existence check'
                )
            }
        )
        return parameters

    def configure(self):
        super().configure()
        self.images_dir = self.get_value_from_config('images_dir')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotations = []
        content_errors = None
        if check_content:
            content_errors = []
            self.images_dir = self.images_dir or self.annotation_file.parent

        original_annotations = read_txt(self.annotation_file)
        num_iterations = len(original_annotations)

        for line_id, line in enumerate(original_annotations):
            identifier, text = line.strip().split(' ')
            annotations.append(CharacterRecognitionAnnotation(identifier, text))
            if check_content:
                if not check_file_existence(self.images_dir / identifier):
                    content_errors.append('{}: does not exist'.format(identifier))
            if progress_callback is not None and line_id % progress_interval:
                progress_callback(line_id / num_iterations * 100)

        label_map = {ind: str(key) for ind, key in enumerate(self.supported_symbols)}
        meta = {'label_map': label_map, 'blank_label': len(label_map)}

        return ConverterReturn(annotations, meta, content_errors)
