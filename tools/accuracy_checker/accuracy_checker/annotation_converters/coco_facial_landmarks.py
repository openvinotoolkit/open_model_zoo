"""
Copyright (c) 2018-2021 Intel Corporation

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
from .format_converter import FileBasedAnnotationConverter, ConverterReturn
from ..representation import Face98LandmarksAnnotation
from ..utils import read_xml, check_file_existence
from ..config import PathField
from xtcocotools.coco import COCO

class COCOFacialLandmarksRecognitionConverter(FileBasedAnnotationConverter):
    __provider__ = 'coco_facial_landmarks'
    annotation_types = (Face98LandmarksAnnotation, )

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

    def _get_mapping_id_name(self, imgs):
        
        id2name = {}
        name2id = {}
        for image_id, image in imgs.items():
            file_name = image['file_name']
            id2name[image_id] = file_name
            name2id[file_name] = image_id

        return id2name, name2id

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        coco = COCO(self.annotation_file)
        img_ids = coco.getImgIds()
        num_images = len(img_ids)
        id2name, self.name2id = self._get_mapping_id_name(coco.imgs)
        num_landmarks = 98
        annotations = []
        scales = []
        centers = []        
        for img_id in img_ids:
            identifier = id2name[img_id]
            ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)          
            objs = coco.loadAnns(ann_ids)
            for obj in objs:
                keypoints = np.array(obj['keypoints']).reshape(-1, 3)
                bbox = obj['bbox']
                landmarks_x, landmarks_y = self.get_landmarks(keypoints, 98)
                landmarks_annotation = Face98LandmarksAnnotation(identifier, np.array(landmarks_x), np.array(landmarks_y))
                landmarks_annotation.metadata['rect'] = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
                annotations.append(landmarks_annotation)
                if progress_callback is not None and img_id % progress_interval == 0:
                    progress_callback(img_id * 100 / len(img_ids))
        return ConverterReturn(annotations, None, None)

    @staticmethod
    def get_landmarks(keypoints, num_landmarks):
        landmarks_x, landmarks_y = np.zeros(num_landmarks), np.zeros(num_landmarks)
        for i, point in enumerate(keypoints):
            
            x, y = point[0], point[1]
            landmarks_x[i] = float(x)
            landmarks_y[i] = float(y)

        return landmarks_x, landmarks_y
