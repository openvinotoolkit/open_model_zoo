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
from ..representation import FacialLandmarks3DAnnotation
from .format_converter import DirectoryBasedAnnotationConverter, ConverterReturn
from ..utils import loadmat


class AFLW20003DConverter(DirectoryBasedAnnotationConverter):
    __provider__ = 'aflw2000_3d'

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        images_list = list(self.data_dir.glob('*.jpg'))
        num_iterations = len(images_list)
        content_errors = [] if check_content else None
        annotations = []
        for img_id, image in enumerate(images_list):
            annotation_file = self.data_dir / image.name.replace('jpg', 'mat')
            if not annotation_file.exists():
                if check_content:
                    content_errors.append('{}: does not exist'.format(annotation_file))
                continue

            image_info = loadmat(annotation_file)
            x_values, y_values, z_values = image_info['pt3d_68']
            x_min, y_min = np.min(x_values), np.min(y_values)
            x_max, y_max = np.max(x_values), np.max(y_values)
            annotation = FacialLandmarks3DAnnotation(image.name, x_values, y_values, z_values)
            annotation.metadata['rect'] = [x_min, y_min, x_max, y_max]
            annotation.metadata['left_eye'] = [36, 39]
            annotation.metadata['right_eye'] = [42, 45]
            annotations.append(annotation)
            if progress_callback is not None and img_id % progress_interval:
                progress_callback(img_id / num_iterations * 100)

        return ConverterReturn(annotations, None, content_errors)
