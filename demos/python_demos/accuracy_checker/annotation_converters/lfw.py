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
from collections import defaultdict
from pathlib import Path

from accuracy_checker.representation.reid_representation import ReIdentificationClassificationAnnotation
from accuracy_checker.utils import check_exists

from .format_converter import BaseFormatConverter


class FaceReidPairwiseConverter(BaseFormatConverter):
    __provider__ = "face_reid_pairwise"

    def convert(self, pairs_file, train_file=None, landmarks_file=None):
        pairs_file = Path(pairs_file).absolute()
        landmarks_map = None
        if landmarks_file:
            landmarks_file = Path(landmarks_file).absolute()
            check_exists(landmarks_file)
            with landmarks_file.open('rt') as landmarks:
                landmarks_list = landmarks.read().split('\n')
                landmarks_map = {}
                for landmark_line in landmarks_list:
                    landmark_line = landmark_line.split('\t')
                    landmarks_map[landmark_line[0]] = [int(point) for point in landmark_line[1:]]
        test_annotations = self.prepare_annotation(pairs_file, True, landmarks_map)
        if train_file:
            train_file = Path(train_file).absolute()
            train_annotations = self.prepare_annotation(train_file, True, landmarks_map)
            test_annotations += train_annotations

        return test_annotations, {}

    @staticmethod
    def get_image_name(person, image_id):
        image_path_pattern = '{}/{}_{}{}.jpg'

        image_path = image_path_pattern.format(person, person, '0'*(4-len(image_id)), image_id)
        return image_path

    def convert_positive(self, pairs, all_images):
        positives = defaultdict(set)
        for data in pairs:
            image1 = self.get_image_name(data[0], data[1])
            image2 = self.get_image_name(data[0], data[2])
            positives[image1].add(image2)
            all_images.add(image1)
            all_images.add(image2)
        return positives, all_images

    def convert_negative(self, pairs, all_images):
        negatives = defaultdict(set)
        for data in pairs:
            image1 = self.get_image_name(data[0], data[1])
            image2 = self.get_image_name(data[2], data[3])
            negatives[image1].add(image2)
            all_images.add(image1)
            all_images.add(image2)
        return negatives, all_images

    def prepare_annotation(self, ann_file, train=False, landmarks_map=None):
        if not ann_file.is_file():
            raise FileNotFoundError("Pairs file {} not found".format(ann_file.as_posix()))
        positive_pairs, negative_pairs = [], []
        with ann_file.open("rt") as f:
            for line in f.readlines()[1:]:  # skip header
                pair = line.strip().split()
                if len(pair) == 3:
                    positive_pairs.append(pair)
                elif len(pair) == 4:
                    negative_pairs.append(pair)
        all_images = set()
        positive_data, all_images = self.convert_positive(positive_pairs, all_images)
        negative_data, all_images = self.convert_negative(negative_pairs, all_images)

        annotations = []
        for image in all_images:
            positive = positive_data[image]
            negative = negative_data[image]
            annotation = ReIdentificationClassificationAnnotation(image, positive, negative)
            if landmarks_map is not None:
                image_landmarks = landmarks_map.get(image)
                annotation.metadata['keypoints'] = image_landmarks
            if train:
                annotation.metadata['train'] = True
            annotations.append(annotation)
        return annotations
