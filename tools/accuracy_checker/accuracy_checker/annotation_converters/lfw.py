"""
Copyright (c) 2018-2020 Intel Corporation

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

from collections import defaultdict
from pathlib import Path

from ..config import PathField
from ..representation import ReIdentificationClassificationAnnotation
from ..utils import read_txt, check_file_existence, OrderedSet

from .format_converter import BaseFormatConverter, ConverterReturn


class LFWConverter(BaseFormatConverter):
    __provider__ = 'lfw'
    annotation_types = (ReIdentificationClassificationAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'pairs_file': PathField(description="Path to file with annotation positive and negative pairs."),
            'landmarks_file': PathField(
                optional=True, description="Path to file with facial landmarks coordinates for annotation images."
            ),
            'images_dir': PathField(
                is_directory=True, optional=True,
                description='path to dataset images, used only for content existence check'
            )
        })

        return configuration_parameters

    def configure(self):
        self.pairs_file = self.get_value_from_config('pairs_file')
        self.landmarks_file = self.get_value_from_config('landmarks_file')
        self.images_dir = self.get_value_from_config('images_dir') or self.pairs_file.parent

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        landmarks_map = {}
        if self.landmarks_file:
            for landmark_line in read_txt(self.landmarks_file):
                landmark_line = landmark_line.split('\t')
                landmarks_map[landmark_line[0]] = [int(point) for point in landmark_line[1:]]

        test_annotations = self.prepare_annotation(
            self.pairs_file, True, landmarks_map, check_content, progress_callback, progress_interval
        )

        return test_annotations

    @staticmethod
    def get_image_name(person, image_id):
        image_path_pattern = '{}/{}_{}{}.jpg'
        return image_path_pattern.format(person, person, '0' * (4 - len(image_id)), image_id)

    def convert_positive(self, pairs, all_images):
        positives = defaultdict(OrderedSet)
        for data in pairs:
            image1 = self.get_image_name(data[0], data[1])
            image2 = self.get_image_name(data[0], data[2])
            positives[image1].add(image2)
            all_images.add(image1)
            all_images.add(image2)

        return positives, all_images

    def convert_negative(self, pairs, all_images):
        negatives = defaultdict(OrderedSet)
        for data in pairs:
            image1 = self.get_image_name(data[0], data[1])
            image2 = self.get_image_name(data[2], data[3])
            negatives[image1].add(image2)
            all_images.add(image1)
            all_images.add(image2)

        return negatives, all_images

    def prepare_annotation(
            self, ann_file: Path, train=False, landmarks_map=None,
            check_content=False, progress_callback=None, progress_interval=100
    ):
        positive_pairs, negative_pairs = [], []
        content_errors = [] if check_content else None
        ann_lines = read_txt(ann_file)
        for line in ann_lines[1:]:  # skip header
            pair = line.strip().split()
            if len(pair) == 3:
                positive_pairs.append(pair)
            elif len(pair) == 4:
                negative_pairs.append(pair)

        all_images = OrderedSet()
        positive_data, all_images = self.convert_positive(positive_pairs, all_images)
        negative_data, all_images = self.convert_negative(negative_pairs, all_images)

        annotations = []
        num_iterations = len(all_images)
        for image_id, image in enumerate(all_images):
            annotation = ReIdentificationClassificationAnnotation(image, positive_data[image], negative_data[image])

            if landmarks_map:
                image_landmarks = landmarks_map.get(image)
                annotation.metadata['keypoints'] = image_landmarks

            if train:
                annotation.metadata['train'] = True

            annotations.append(annotation)

            if check_content:
                image_full_path = self.images_dir / image
                if not check_file_existence(image_full_path):
                    content_errors.append('{}: does nit exist'.format(image_full_path))

            if progress_callback is not None and image_id % progress_interval == 0:
                progress_callback(image_id / num_iterations * 100)

        return ConverterReturn(annotations, None, content_errors)
