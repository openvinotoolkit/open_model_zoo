"""
Copyright (c) 2018-2024 Intel Corporation

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
import json
import numpy as np
from .format_converter import FileBasedAnnotationConverter, ConverterReturn
from ..representation import FacialLandmarksHeatMapAnnotation
from ..config import PathField


class COCOFacialLandmarksRecognitionConverter(FileBasedAnnotationConverter):
    __provider__ = 'coco_facial_landmarks'
    annotation_types = (FacialLandmarksHeatMapAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'images_dir': PathField(
                is_directory=True, optional=True,
                description='path to dataset images, used only for content existence check'
            )
        })
        return configuration_parameters

    def configure(self):
        super().configure()
        self.images_dir = self.get_value_from_config('images_dir') or self.annotation_file.parent

    @staticmethod
    def _collect_image_ids(data):
        result = {}
        for itm in data:
            img_name = itm["file_name"]
            img_id = itm["id"]
            result[img_id] = img_name

        return result

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        with open(self.annotation_file, encoding='UTF-8') as f:
            data = json.load(f)
        coco_ann = data["annotations"]
        id2name = self._collect_image_ids(data["images"])
        num_landmarks = 98
        annotations = []
        for ann_id, ann in enumerate(coco_ann[1:]):
            identifier = id2name[ann["image_id"]]
            bbox = ann["bbox"]
            keypoints = np.array(ann["keypoints"]).reshape(-1, 3)
            landmarks_x, landmarks_y = self.get_landmarks(keypoints, num_landmarks)
            landmarks_annotation = FacialLandmarksHeatMapAnnotation(identifier,
                                                                    np.array(landmarks_x),
                                                                    np.array(landmarks_y))
            landmarks_annotation.metadata['rect'] = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
            annotations.append(landmarks_annotation)
            if progress_callback is not None and ann_id % progress_interval == 0:
                progress_callback(ann_id * 100 / len(coco_ann[1:]))
        return ConverterReturn(annotations, None, None)

    @staticmethod
    def get_landmarks(keypoints, num_landmarks):
        landmarks_x, landmarks_y = np.zeros(num_landmarks), np.zeros(num_landmarks)
        for i, point in enumerate(keypoints):
            x, y = point[0], point[1]
            landmarks_x[i] = float(x)
            landmarks_y[i] = float(y)

        return landmarks_x, landmarks_y
