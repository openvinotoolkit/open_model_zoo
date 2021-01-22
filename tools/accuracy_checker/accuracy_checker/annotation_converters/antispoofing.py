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

from .format_converter import ConverterReturn, FileBasedAnnotationConverter
from ..representation import ClassificationAnnotation
from ..utils import read_json, check_file_existence
from ..config import PathField, NumberField

class AntispoofingDatasetConverter(FileBasedAnnotationConverter):
    __provider__ = 'antispoofing'
    annotation_types = (ClassificationAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update(
            {
                'data_dir': PathField(
                    is_directory=True, optional=False,
                    description='path to input images'
                ),
                'dataset_meta_file': PathField(
                    description='path to json file with dataset meta (e.g. label_map)', optional=True
                ),
                'label_id': NumberField(
                    description='number of label in the annotation file representing spoof/real labels',
                    optional=True, default=43, value_type=int
                ),
                'annotation_file': PathField(
                    description='path to json file with dataset annotations'
                    '({index : {path: ..., labels: ..., boxes: ... (optional)}})', optional=False
                )
            }
        )
        return configuration_parameters

    def configure(self):
        super().configure()
        self.data_dir = self.get_value_from_config('data_dir')
        self.annotations = self.get_value_from_config('annotation_file')
        self.label_id = self.get_value_from_config('label_id')
        self.meta = self.get_value_from_config('dataset_meta_file')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        """Reads data from disk and returns dataset in converted for AC format

        Args:
            check_content (bool, optional): Check if content is valid. Defaults to False.
            progress_callback (bool, optional): Display progress. Defaults to None.
            progress_interval (int, optional): Units to display progress. Defaults to 100 (percent).

        Returns:
            [type]: Converted dataset
        """
        annotation_tuple = self.generate_annotations()
        annotations = []
        content_errors = None if not check_content else []
        meta = self.generate_meta()
        num_iterations = len(annotations)

        for i, (img_name, label, bbox) in enumerate(annotation_tuple):
            image_annotation = ClassificationAnnotation(img_name, label)
            if bbox:
                image_annotation.metadata['rect'] = bbox
            annotations.append(image_annotation)

            if check_content:
                if not check_file_existence(self.data_dir /img_name):
                    content_errors.append('{}: does not exist'.format(img_name))
            if progress_callback is not None and i % progress_interval == 0:
                progress_callback(i / num_iterations * 100)

        return ConverterReturn(annotations, meta, content_errors)

    def generate_meta(self):
        if not self.meta:
            return {'label_map': {'real': 0, 'spoof': 1}}
        dataset_meta = read_json(self.meta)
        label_map = dataset_meta.get('label_map')
        dataset_meta['label_map'] = label_map or {'real': 0, 'spoof': 1}
        return dataset_meta

    def generate_annotations(self):
        ''' read json file with images paths and return
        list of the items'''
        annotation_store = []
        dataset_annotations = read_json(self.annotations)
        for index in dataset_annotations:
            path = dataset_annotations[index]['path']
            target_label = dataset_annotations[index]['labels'][self.label_id]
            bbox = dataset_annotations[index].get('bbox')
            annotation_store.append((path, target_label, bbox))

        return annotation_store
