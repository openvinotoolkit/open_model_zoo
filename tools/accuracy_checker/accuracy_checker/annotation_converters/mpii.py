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

import numpy as np
from .format_converter import FileBasedAnnotationConverter, ConverterReturn
from ..config import PathField
from ..utils import read_json
from ..representation import PoseEstimationAnnotation

joints = {
    'head': 9,
    'lsho': 13,
    'lelb': 14,
    'lwri': 15,
    'lhip': 3,
    'lkne': 4,
    'lank': 5,
    'rsho': 12,
    'relb': 11,
    'rwri': 10,
    'rkne': 1,
    'rank': 0,
    'rhip': 2
}


class MPIIDatasetConverter(FileBasedAnnotationConverter):
    __provider__ = 'mpii'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'headboxes_file': PathField(),
        })
        return params

    def configure(self):
        super().configure()
        self.headboxes_file = self.get_value_from_config('headboxes_file')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotations = []
        headboxes = np.load(str(self.headboxes_file))

        valid_annotations = [ann for ann in read_json(self.annotation_file) if ann['isValidation']]
        num_iterations = len(valid_annotations)
        content_errors = [] if check_content else None
        for idx, ann in enumerate(valid_annotations):
            identifier = ann['img_paths']
            points = np.array(ann['joint_self'])
            center = ann['objpos']
            scale = float(ann['scale_provided'])
            x_values, y_values, vis = points.T # pylint: disable=E0633
            annotation = PoseEstimationAnnotation(identifier, x_values, y_values, visibility=vis)
            annotation.metadata['center'] = center
            annotation.metadata['scale'] = scale
            annotation.metadata['headbox'] = headboxes[:, :, idx]
            annotations.append(annotation)
            if progress_callback is not None and idx % progress_interval == 0:
                progress_callback(idx * 100 / num_iterations)

        return ConverterReturn(annotations, {'joints': joints}, content_errors)
