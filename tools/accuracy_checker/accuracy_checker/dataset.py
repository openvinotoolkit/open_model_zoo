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

from copy import deepcopy
from pathlib import Path
import warnings

from .annotation_converters import BaseFormatConverter, save_annotation, make_subset, analyze_dataset
from .config import (
    ConfigValidator, StringField, PathField, ListField,
    DictField, BaseField, NumberField, ConfigError, BoolField
)
from .utils import JSONDecoderWithAutoConversion, read_json, get_path, contains_all, set_image_metadata, OrderedSet
from .representation import BaseRepresentation, ReIdentificationClassificationAnnotation, ReIdentificationAnnotation
from .data_readers import DataReaderField, REQUIRES_ANNOTATIONS


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
    shuffle = BoolField(optional=True)
    subsample_seed = NumberField(value_type=int, min_value=0, optional=True)
    analyze_dataset = BaseField(optional=True)
    segmentation_masks_source = PathField(is_directory=True, optional=True)
    additional_data_source = PathField(is_directory=True, optional=True)
    batch = NumberField(value_type=int, min_value=1, optional=True)


class Dataset:
    def __init__(self, config_entry, delayed_annotation_loading=False):
        self._config = config_entry
        self.batch = self.config.get('batch')
        self.iteration = 0
        dataset_config = DatasetConfig('Dataset')
        dataset_config.validate(self._config)
        if not delayed_annotation_loading:
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
        if subsample_size is not None:
            subsample_seed = self._config.get('subsample_seed', 666)
            shuffle = self._config.get('shuffle', True)

            annotation = create_subset(annotation, subsample_size, subsample_seed, shuffle)

        if self._config.get('analyze_dataset', False):
            if self._config.get('segmentation_masks_source'):
                meta['segmentation_masks_source'] = self._config.get('segmentation_masks_source')
            meta = analyze_dataset(annotation, meta)
            if meta.get('segmentation_masks_source'):
                del meta['segmentation_masks_source']

        if use_converted_annotation and contains_all(self._config, ['annotation', 'annotation_conversion']):
            annotation_name = self._config['annotation']
            meta_name = self._config.get('dataset_meta')
            if meta_name:
                meta_name = Path(meta_name)
            save_annotation(annotation, meta, Path(annotation_name), meta_name)

        self._annotation = annotation
        self._meta = meta or {}
        self.name = self._config.get('name')
        self.subset = None

    @property
    def annotation(self):
        return self._annotation

    @property
    def config(self):
        return deepcopy(self._config) #read-only

    @property
    def identifiers(self):
        return [ann.identifier for ann in self.annotation]

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
        batch_input_ids, batch_annotation = self.__getitem__(self.iteration)
        self.iteration += 1
        context.annotation_batch = batch_annotation
        context.identifiers_batch = [annotation.identifier for annotation in batch_annotation]
        context.input_ids_batch = batch_input_ids

    def __getitem__(self, item):
        if self.batch is None:
            self.batch = 1
        if self.size <= item * self.batch:
            raise IndexError

        batch_start = item * self.batch
        batch_end = min(self.size, batch_start + self.batch)
        if self.subset:
            batch_ids = self.subset[batch_start:batch_end]
            return batch_ids, [self._annotation[idx] for idx in batch_ids]
        batch_ids = range(batch_start, batch_end)

        return batch_ids, self._annotation[batch_start:batch_end]

    def make_subset(self, ids=None, start=0, step=1, end=None, accept_pairs=False):
        pairwise_subset = isinstance(
            self._annotation[0], (ReIdentificationAnnotation, ReIdentificationClassificationAnnotation)
        )
        if ids:
            self.subset = ids if not pairwise_subset else self._make_subset_pairwise(ids, accept_pairs)
            return
        if not end:
            end = self.size
        ids = range(start, end, step)
        self.subset = ids if not pairwise_subset else self._make_subset_pairwise(ids, accept_pairs)

    def _make_subset_pairwise(self, ids, add_pairs=False):
        subsample_set = OrderedSet()
        pairs_set = OrderedSet()
        if isinstance(self._annotation[0], ReIdentificationClassificationAnnotation):
            identifier_to_index = {annotation.identifier: index for index, annotation in enumerate(self._annotation)}
            for idx in ids:
                subsample_set.add(idx)
                current_annotation = self._annotation[idx]
                positive_pairs = [
                    identifier_to_index[pair_identifier] for pair_identifier in current_annotation.positive_pairs
                ]
                pairs_set |= positive_pairs
                negative_pairs = [
                    identifier_to_index[pair_identifier] for pair_identifier in current_annotation.positive_pairs
                ]
                pairs_set |= negative_pairs
        else:
            for idx in ids:
                subsample_set.add(idx)
                selected_annotation = self._annotation[idx]
                if not selected_annotation.query:
                    query_for_person = [
                        idx for idx, annotation in enumerate(self._annotation)
                        if annotation.person_id == selected_annotation.person_id and annotation.query
                    ]
                    pairs_set |= OrderedSet(query_for_person)
                else:
                    gallery_for_person = [
                        idx for idx, annotation in enumerate(self._annotation)
                        if annotation.person_id == selected_annotation.person_id and not annotation.query
                    ]
                    pairs_set |= OrderedSet(gallery_for_person)

        if add_pairs:
            subsample_set |= pairs_set

        return list(subsample_set)

    def set_annotation_metadata(self, annotation, image, data_source):
        set_image_metadata(annotation, image)
        annotation.set_data_source(data_source)
        segmentation_mask_source = self.config.get('segmentation_masks_source')
        annotation.set_segmentation_mask_source(segmentation_mask_source)
        annotation.set_additional_data_source(self.config.get('additional_data_source'))
        annotation.set_dataset_metadata(self.metadata)

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

    def reset(self, reload_annotation=False):
        self.subset = None
        if reload_annotation:
            self._load_annotation()

    def set_annotation(self, annotation):
        subsample_size = self._config.get('subsample_size')
        if subsample_size is not None:
            subsample_seed = self._config.get('subsample_seed', 666)

            annotation = create_subset(annotation, subsample_size, subsample_seed)

        if self._config.get('analyze_dataset', False):
            if self._config.get('segmentation_masks_source'):
                self.metadata['segmentation_masks_source'] = self._config.get('segmentation_masks_source')
            self.metadata = analyze_dataset(annotation, self.metadata)
            if self.metadata.get('segmentation_masks_source'):
                del self.metadata['segmentation_masks_source']

        self._annotation = annotation
        self.name = self._config.get('name')
        self.subset = None

    def provide_data_info(self, reader, annotations):
        for ann in annotations:
            input_data = reader(ann.identifier)
            self.set_annotation_metadata(ann, input_data, reader.data_source)
        return annotations


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


def create_subset(annotation, subsample_size, subsample_seed, shuffle=True):
    if isinstance(subsample_size, str):
        if subsample_size.endswith('%'):
            try:
                subsample_size = float(subsample_size[:-1])
            except ValueError:
                raise ConfigError('invalid value for subsample_size: {}'.format(subsample_size))
            if subsample_size <= 0:
                raise ConfigError('subsample_size should be > 0')
            subsample_size *= len(annotation) / 100
            subsample_size = int(subsample_size) or 1
    try:
        subsample_size = int(subsample_size)
    except ValueError:
        raise ConfigError('invalid value for subsample_size: {}'.format(subsample_size))
    if subsample_size < 1:
        raise ConfigError('subsample_size should be > 0')
    return make_subset(annotation, subsample_size, subsample_seed, shuffle)


class DatasetWrapper:
    def __init__(self, data_reader, annotation_reader=None, tag='', dataset_config=None):
        self.tag = tag
        self.data_reader = data_reader
        self.annotation_reader = annotation_reader
        self._batch = 1 if not annotation_reader else annotation_reader.batch
        self.subset = None
        self.dataset_config = dataset_config or {}
        if not annotation_reader:
            self._identifiers = [file.name for file in self.data_reader.data_source.glob('*')]

    def __getitem__(self, item):
        if self.batch is None:
            self.batch = 1
        if self.size <= item * self.batch:
            raise IndexError
        batch_annotation = []
        if self.annotation_reader:
            batch_annotation_ids, batch_annotation = self.annotation_reader[item]
            batch_identifiers = [annotation.identifier for annotation in batch_annotation]
            batch_input = [self.data_reader(identifier=identifier) for identifier in batch_identifiers]
            for annotation, input_data in zip(batch_annotation, batch_input):
                set_image_metadata(annotation, input_data)
                annotation.set_data_source(self.data_reader.data_source)
                segmentation_mask_source = self.annotation_reader.config.get('segmentation_masks_source')
                annotation.set_segmentation_mask_source(segmentation_mask_source)
                annotation.set_additional_data_source(self.annotation_reader.config.get('additional_data_source'))
            return batch_annotation_ids, batch_annotation, batch_input, batch_identifiers
        batch_start = item * self.batch
        batch_end = min(self.size, batch_start + self.batch)
        batch_input_ids = self.subset[batch_start:batch_end] if self.subset else range(batch_start, batch_end)
        batch_identifiers = [self._identifiers[idx] for idx in batch_input_ids]
        batch_input = [self.data_reader(identifier=identifier) for identifier in batch_identifiers]

        return batch_input_ids, batch_annotation, batch_input, batch_identifiers

    def __len__(self):
        if self.annotation_reader:
            return self.annotation_reader.size
        if self.subset:
            return len(self.subset)
        return len(self._identifiers)

    def make_subset(self, ids=None, start=0, step=1, end=None, accept_pairs=False):
        if self.annotation_reader:
            self.annotation_reader.make_subset(ids, start, step, end, accept_pairs)
        if ids:
            self.subset = ids
            return
        if not end:
            end = self.size
        self.subset = range(start, end, step)
        if self.data_reader.name in REQUIRES_ANNOTATIONS:
            self.data_reader.subset = self.subset

    @property
    def batch(self):
        return self._batch

    @batch.setter
    def batch(self, batch):
        if self.annotation_reader:
            self.annotation_reader.batch = batch
        self._batch = batch

    def reset(self, reload_annotation=False):
        if self.subset:
            self.subset = None
        if self.annotation_reader:
            self.annotation_reader.reset(reload_annotation)
        self.data_reader.reset()

    @property
    def full_size(self):
        if self.annotation_reader:
            return self.annotation_reader.full_size
        return len(self._identifiers)

    @property
    def size(self):
        return self.__len__()

    @property
    def multi_infer(self):
        return getattr(self.data_reader, 'multi_infer', False)
