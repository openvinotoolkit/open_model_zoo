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

from .format_converter import FileBasedAnnotationConverter, ConverterReturn, verify_label_map
from ..utils import read_txt, check_file_existence
from ..representation import SegmentationAnnotation
from ..config import PathField
from ..utils import read_json


class CamVidConverter(FileBasedAnnotationConverter):
    __provider__ = 'camvid'
    annotation_type = (SegmentationAnnotation, )
    meta = {
        'label_map': {
            0: 'Sky',
            1: 'Building',
            2: 'Pole',
            3: 'Road',
            4: 'Pavement',
            5: 'Tree',
            6: 'SignSymbol',
            7: 'Fence',
            8: 'Car',
            9: 'Pedestrian',
            10: 'Bicyclist',
            11: 'Unlabelled'
        },
        'background_label': 11,
        'segmentation_colors': (
            (128, 128, 128), (128, 0, 0), (192, 192, 128), (128, 64, 128), (60, 40, 222), (128, 128, 0),
            (192, 128, 128), (64, 64, 128), (64, 0, 128), (64, 64, 0), (0, 128, 192), (0, 0, 0)
        )
    }

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'dataset_meta_file': PathField(
                description='path to json file with dataset meta (e.g. label_map, color_encoding', optional=True
            )})
        return params

    def configure(self):
        super().configure()
        self.dataset_meta = self.get_value_from_config('dataset_meta_file')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotation = read_txt(self.annotation_file)
        annotations = []
        content_errors = None if not check_content else []
        num_iterations = len(annotation)
        for line_id, line in enumerate(annotation):
            image_path, gt_path = line.split(' ')
            if check_content:
                if not check_file_existence(image_path):
                    content_errors.append("{}: does not exists".format(image_path))
                if not check_file_existence(gt_path):
                    content_errors.append('{}: does not exist'.format(gt_path))
            identifier = '/'.join(image_path.split('/')[-2:])
            gt_file = '/'.join(gt_path.split('/')[-2:])
            annotations.append(SegmentationAnnotation(identifier, gt_file))
            if progress_callback is not None and line_id % progress_interval == 0:
                progress_callback(line_id * 100 / num_iterations)
        meta = self.meta
        if self.dataset_meta:
            meta = read_json(self.dataset_meta)
            if 'label_map' in meta:
                meta['label_map'] = verify_label_map(meta['label_map'])
            if 'labels' in meta and 'label_map' not in meta:
                meta['label_map'] = dict(enumerate(meta['labels']))

        return ConverterReturn(annotations, meta, content_errors)
