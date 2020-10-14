""""
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

from .format_converter import BaseFormatConverter, ConverterReturn
from ..utils import loadmat, UnsupportedPackage, check_file_existence
from ..representation import PlaceRecognitionAnnotation
from ..config import PathField, NumberField
try:
    from sklearn.neighbors import NearestNeighbors
except ImportError as import_error:
    NearestNeighbors = UnsupportedPackage('sklearn.neighbors', import_error)


class PlaceRecognitionDatasetConverter(BaseFormatConverter):
    __provider__ = 'place_recognition'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'split_file': PathField(),
            'images_dir': PathField(is_directory=True, optional=True),
            'distance_threshold': NumberField(optional=True, default=25, min_value=0)
        })
        return params

    def configure(self):
        self.split_file = self.get_value_from_config('split_file')
        self.data_dir = self.get_value_from_config('images_dir') or self.split_file.parent
        self.distance = self.get_value_from_config('distance_therhold')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        queries, loc_query, gallery, loc_gallery = self.read_db()
        positives = self.get_positives(loc_gallery, loc_query)
        annotations = []
        num_iterations = len(queries) + len(gallery)
        content_errors = None if not check_content else []
        for idx, image in enumerate(gallery):
            positive_q = positives[idx]
            if check_content and not check_file_existence(self.data_dir / image):
                content_errors.append('{}: des not exist'.format(self.data_dir / image))
            annotations.append(PlaceRecognitionAnnotation(image, positive_q, False))
            if progress_callback and idx % progress_interval == 0:
                progress_callback(idx * 100 / num_iterations)
        for idx, image in enumerate(queries):
            image = 'queries_real/'+image
            if check_content and not check_file_existence(self.data_dir / image):
                content_errors.append('{}: des not exist'.format(self.data_dir / image))
            annotations.append(PlaceRecognitionAnnotation(image, idx, True))
            if progress_callback and (idx + len(gallery)) % progress_callback == 0:
                progress_callback((idx + len(gallery)) * 100 / num_iterations)

        return ConverterReturn(annotations, None, content_errors)

    def read_db(self):
        data = loadmat(str(self.split_file))['dbStruct']
        db_image = [f[0] for f in data['dbImageFns']]
        utm_db = np.array(data['utmDb']).T

        q_image = [f[0] for f in data['qImageFns']]
        utm_q = np.array(data['utmQ']).T
        return q_image, utm_q, db_image, utm_db

    def get_positives(self, query_loc, gallery_loc):
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(gallery_loc)
        _, positives = knn.radius_neighbors(query_loc, radius=self.distance)

        return positives
