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

from .format_converter import FileBasedAnnotationConverter, BaseFormatConverter, ConverterReturn, verify_label_map
from ..utils import read_txt, check_file_existence
from ..representation import SegmentationAnnotation
from ..config import PathField, NumberField
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
                    content_errors.append("{}: does not exist".format(image_path))
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


class CamVid32DatasetConverter(BaseFormatConverter):
    __provider__ = 'camvid_32'
    meta = {
        'label_map': {
            0: 'Animal',
            1: 'Archway',
            2: 'Bicyclist',
            3: 'Bridge',
            4: 'Building',
            5: 'Car',
            6: 'CartLuggagePram',
            7: 'Child',
            8: 'Column_Pole',
            9: 'Fence',
            10: 'LaneMkgsDriv',
            11: 'LaneMkgsNonDriv',
            12: 'Misc_Text',
            13: 'MotorcycleScooter',
            14: 'OtherMoving',
            15: 'ParkingBlock',
            16: 'Pedestrian',
            17: 'Road',
            18: 'RoadShoulder',
            19: 'Sidewalk',
            20: 'SignSymbol',
            21: 'Sky',
            22: 'SUVPickupTruck',
            23: 'TrafficCone',
            24: 'TrafficLight',
            25: 'Train',
            26: 'Tree',
            27: 'Truck_Bus',
            28: 'Tunnel',
            29: 'VegetationMisc',
            30: 'Void',
            31: 'Wall'
        },
        'background_label': 30,
        'segmentation_colors': (
            (64, 128, 64), (192, 0, 128), (0, 128, 192), (0, 128, 64), (128, 0, 0), (64, 0, 128), (64, 0, 192),
            (192, 128, 64), (192, 192, 128), (64, 64, 128), (128, 0, 192), (192, 0, 64), (128, 128, 64), (192, 0, 192),
            (128, 64, 64), (64, 192, 128), (64, 64, 0), (128, 64, 128), (128, 128, 192), (0, 0, 192), (192, 128, 128),
            (128, 128, 128), (64, 128, 192), (0, 0, 64), (0, 64, 64), (192, 64, 128), (128, 128, 0), (192, 128, 192),
            (64, 0, 64), (192, 192, 0), (0, 0, 0), (64, 192, 0)
        )
    }

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            "labels_dir": PathField(is_directory=True, description='path to directory with labeled images'),
            'images_dir': PathField(is_directory=True, description='path to directory with input images'),
            'val_subset_ratio': NumberField(
                value_type=float, min_value=0, max_value=1, default=1, description='subset ration for validation'
            ),
            'dataset_meta_file': PathField(
                optional=True, description='path to json file with dataset meta (e.g. label_map, color_encoding')
        })
        return params

    def configure(self):
        self.labels_dir = self.get_value_from_config('labels_dir')
        self.images_dir = self.get_value_from_config('images_dir')
        self.val_subset_ratio = self.get_value_from_config('val_subset_ratio')
        self.dataset_meta = self.get_value_from_config('dataset_meta_file')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        label_files = list(self.labels_dir.glob('*.png'))
        annotations = []
        val_subset_size = int(len(label_files) * self.val_subset_ratio)
        val_labels = label_files[len(label_files)-val_subset_size:]
        content_errors = None if not check_content else []
        for idx, label in enumerate(val_labels):
            identifier = label.name.replace('_L', '')
            if check_content:
                if not check_file_existence(self.images_dir / identifier):
                    content_errors.append('{}: does not exist'.format(self.images_dir / identifier))
            annotations.append(SegmentationAnnotation(identifier, label.name))
            if progress_callback and idx % progress_interval == 0:
                progress_callback(idx * 100 / val_subset_size)

        meta = self.meta
        if self.dataset_meta:
            meta = read_json(self.dataset_meta)
            if 'label_map' in meta:
                meta['label_map'] = verify_label_map(meta['label_map'])
            if 'labels' in meta and 'label_map' not in meta:
                meta['label_map'] = dict(enumerate(meta['labels']))

        return ConverterReturn(annotations, meta, content_errors)
