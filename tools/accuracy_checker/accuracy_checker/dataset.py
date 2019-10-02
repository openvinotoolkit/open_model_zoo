"""
Copyright (c) 2019 Intel Corporation

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

from copy import deepcopy
from pathlib import Path
import warnings

from .annotation_converters import BaseFormatConverter, save_annotation, make_subset, analyze_dataset
from .config import ConfigValidator, StringField, PathField, ListField, DictField, BaseField, NumberField, ConfigError
from .utils import JSONDecoderWithAutoConversion, read_json, get_path, contains_all, set_image_metadata
from .representation import BaseRepresentation
from .data_readers import DataReaderField


class DatasetConfig(ConfigValidator):
    """
    Specifies configuration structure for dataset
    """
    name = StringField()
    annotation = PathField(optional=True, check_exists=False)
    data_source = PathField(optional=True, check_exists=False)
    dataset_meta = PathField(optional=True, check_exists=False)
    metrics = ListField(allow_empty=False, optional=True)
    postprocessing = ListField(allow_empty=False, optional=True)
    preprocessing = ListField(allow_empty=False, optional=True)
    reader = DataReaderField(optional=True)
    annotation_conversion = DictField(optional=True)
    subsample_size = BaseField(optional=True)
    subsample_seed = NumberField(value_type=int, min_value=0, optional=True)
    analyze_dataset = BaseField(optional=True)


class Dataset:
    def __init__(self, config_entry):
        self._config = config_entry
        self.batch = 1
        self.iteration = 0
        dataset_config = DatasetConfig('Dataset')
        dataset_config.validate(self._config)
        self._images_dir = Path(self._config.get('data_source', ''))
        self._load_annotation()

    def _load_annotation(self):
        annotation, meta = None, None
        use_converted_annotation = True
        if 'annotation' in self._config:
            annotation_file = Path(self._config['annotation'])
            if annotation_file.exists():
                annotation = read_annotation(get_path(annotation_file))
                meta = self._load_meta()
                use_converted_annotation = False
        if not annotation and 'annotation_conversion' in self._config:
            annotation, meta = self._convert_annotation()

        if not annotation:
            raise ConfigError('path to converted annotation or data for conversion should be specified')

        subsample_size = self._config.get('subsample_size')
        if subsample_size:
            subsample_seed = self._config.get('subsample_seed', 666)
            if isinstance(subsample_size, str):
                if subsample_size.endswith('%'):
                    subsample_size = float(subsample_size[:-1]) / 100 * len(annotation)
            subsample_size = int(subsample_size)
            annotation = make_subset(annotation, subsample_size, subsample_seed)

        if self._config.get('analyze_dataset', False):
            analyze_dataset(annotation, meta)

        if use_converted_annotation and contains_all(self._config, ['annotation', 'annotation_conversion']):
            annotation_name = self._config['annotation']
            meta_name = self._config.get('dataset_meta')
            if meta_name:
                meta_name = Path(meta_name)
            save_annotation(annotation, meta, Path(annotation_name), meta_name)

        self._annotation = annotation
        self._meta = meta
        self.name = self._config.get('name')
        self.subset = None

    @property
    def annotation(self):
        return self._annotation

    @property
    def config(self):
        return deepcopy(self._config) #read-only

    def __len__(self):
        if self.subset:
            return len(self.subset)
        return len(self._annotation)

    @property
    def metadata(self):
        return deepcopy(self._meta) #read-only

    @property
    def labels(self):
        return self._meta.get('label_map', {})

    @property
    def size(self):
        return self.__len__()

    @property
    def full_size(self):
        return len(self._annotation)

    def __call__(self, context, *args, **kwargs):
        batch_annotation = self.__getitem__(self.iteration)
        self.iteration += 1
        context.annotation_batch = batch_annotation
        context.identifiers_batch = [annotation.identifier for annotation in batch_annotation]

    def __getitem__(self, item):
        if self.size <= item * self.batch:
            raise IndexError

        batch_start = item * self.batch
        batch_end = min(self.size, batch_start + self.batch)
        if self.subset:
            return [self._annotation[idx] for idx in self.subset[batch_start:batch_end]]

        return self._annotation[batch_start:batch_end]

    def make_subset(self, ids=None, start=0, step=1, end=None):
        if ids:
            self.subset = ids
            return
        if not end:
            end = self.size
        self.subset = range(start, end, step)

    @staticmethod
    def set_image_metadata(annotation, images):
        image_sizes = []
        data = images.data
        if not isinstance(data, list):
            data = [data]
        for image in data:
            image_sizes.append(image.shape)
        annotation.set_image_size(image_sizes)

    def set_annotation_metadata(self, annotation, image, data_source):
        self.set_image_metadata(annotation, image.data)
        annotation.set_data_source(data_source)

    def _load_meta(self):
        meta_data_file = self._config.get('dataset_meta')
        return read_json(meta_data_file, cls=JSONDecoderWithAutoConversion) if meta_data_file else None

    def _convert_annotation(self):
        conversion_params = self._config.get('annotation_conversion')
        converter = conversion_params['converter']
        annotation_converter = BaseFormatConverter.provide(converter, conversion_params)
        results = annotation_converter.convert()
        annotation = results.annotations
        meta = results.meta
        errors = results.content_check_errors
        if errors:
            warnings.warn('Following problems were found during conversion:\n{}'.format('\n'.join(errors)))

        return annotation, meta

    def reset(self):
        self.subset = None
        self._load_annotation()


def read_annotation(annotation_file: Path):
    annotation_file = get_path(annotation_file)

    result = []
    with annotation_file.open('rb') as file:
        while True:
            try:
                result.append(BaseRepresentation.load(file))
            except EOFError:
                break

    return result


class DatasetWrapper:
    def __init__(self, data_reader, annotation_reader=None, tag=''):
        self.tag = tag
        self.data_reader = data_reader
        self.annotation_reader = annotation_reader
        self._batch = 1
        self.subset = None
        if not annotation_reader:
            self._identifiers = [file.name for file in self.data_reader.data_source.glob('*')]

    def __getitem__(self, item):
        if self.size <= item * self.batch:
            raise IndexError
        batch_annotation = []
        if self.annotation_reader:
            batch_annotation = self.annotation_reader[item]
            batch_identifiers = [annotation.identifier for annotation in batch_annotation]
            batch_input = [self.data_reader(identifier=identifier) for identifier in batch_identifiers]
            for annotation, input_data in zip(batch_annotation, batch_input):
                set_image_metadata(annotation, input_data)
                annotation.metadata['data_source'] = self.data_reader.data_source
            return batch_annotation, batch_input, batch_identifiers
        batch_start = item * self.batch
        batch_end = min(self.size, batch_start + self.batch)
        if self.subset:
            batch_identifiers = [self._identifiers[idx] for idx in self.subset[batch_start:batch_end]]
        else:
            batch_identifiers = self._identifiers[batch_start:batch_end]
        batch_input = [self.data_reader(identifier=identifier) for identifier in batch_identifiers]

        return batch_annotation, batch_input, batch_identifiers

    def __len__(self):
        if self.annotation_reader:
            return self.annotation_reader.size
        if self.subset:
            return len(self.subset)
        return len(self._identifiers)

    def make_subset(self, ids=None, start=0, step=1, end=None):
        if self.annotation_reader:
            self.annotation_reader.make_subset(ids, start, step, end)
        if ids:
            self.subset = ids
            return
        if not end:
            end = self.size
        self.subset = range(start, end, step)

    @property
    def batch(self):
        return self._batch

    @batch.setter
    def batch(self, batch):
        if self.annotation_reader:
            self.annotation_reader.batch = batch
        self._batch = batch

    def reset(self):
        if self.subset:
            self.subset = None
        if self.annotation_reader:
            self.annotation_reader.reset()

    @property
    def full_size(self):
        if self.annotation_reader:
            return self.annotation_reader.full_size
        return len(self._identifiers)

    @property
    def size(self):
        return self.__len__()
