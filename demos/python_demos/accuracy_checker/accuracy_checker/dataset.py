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

from pathlib import Path

from .config import ConfigValidator, StringField, PathField, ListField
from .utils import JSONDecoderWithAutoConversion, read_annotation, read_json
from .data_readers import BaseReader

class DatasetConfig(ConfigValidator):
    """
    Specifies configuration structure for dataset
    """
    name = StringField()
    annotation = PathField(check_exists=True)
    data_source = PathField(check_exists=True)
    dataset_meta = PathField(check_exists=True, optional=True)
    metrics = ListField(allow_empty=False)
    postprocessing = ListField(allow_empty=False, optional=True)
    preprocessing = ListField(allow_empty=False, optional=True)
    reader = StringField(optional=True, choices=BaseReader.providers.keys())


class DataRepresentation:
    def __init__(self, data, meta=None):
        self.data = data
        self.metadata = meta or {}


class Dataset:
    def __init__(self, config_entry, preprocessor):
        self._config = config_entry
        self._preprocessor = preprocessor

        self.batch = 1

        dataset_config = DatasetConfig('Dataset')
        dataset_config.validate(self._config)

        self._images_dir = Path(self._config.get('data_source', ''))
        self._annotation = read_annotation(self._config['annotation'])
        self._meta = self._load_meta()
        self.size = len(self._annotation)
        self.name = self._config.get("name")

        data_reader = self._config.get("reader", "opencv_imread").lower()

        self.read_image_fn = BaseReader.provide(data_reader)

    @property
    def annotation(self):
        return self._annotation

    def __len__(self):
        return self.size

    @property
    def metadata(self):
        return self._meta

    @property
    def labels(self):
        return self._meta.get("label_map", {})

    def __getitem__(self, item):
        if self.size <= item * self.batch:
            raise IndexError

        batch_start = item * self.batch
        batch_end = min(self.size, batch_start + self.batch)
        batch_annotation = self._annotation[batch_start:batch_end]

        identifiers = [annotation.identifier for annotation in batch_annotation]
        paths = self._get_image_paths(identifiers)
        images = self._read_images(paths)

        for image, annotation in zip(images, batch_annotation):
            self.set_annotation_metadata(annotation, image.data, self._images_dir)

        preprocessed = self._preprocessor.process(images, batch_annotation)

        return batch_annotation, preprocessed

    @staticmethod
    def set_image_metadata(annotation, image):
        h, w, c = image.shape
        annotation.set_image_size(h, w, c)

    def set_annotation_metadata(self, annotation, image, data_source):
        self.set_image_metadata(annotation, image)
        annotation.set_data_source(data_source)

    def _get_image_paths(self, image_identifiers):
        return [self._images_dir / img for img in image_identifiers]

    def _read_images(self, paths):
        images = []
        for path in paths:
            images.append(DataRepresentation(self.read_image_fn(path)))

        return images

    def _load_meta(self):
        meta_data_file = self._config.get('dataset_meta')

        return read_json(meta_data_file, cls=JSONDecoderWithAutoConversion) if meta_data_file else None
