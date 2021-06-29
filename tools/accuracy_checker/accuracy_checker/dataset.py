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

from copy import deepcopy
from pathlib import Path
from collections import OrderedDict
import warnings
import pickle
import numpy as np
import yaml

from .annotation_converters import (
    BaseFormatConverter, DatasetConversionInfo, save_annotation, make_subset, analyze_dataset
)
from .metrics import  Metric
from .preprocessor import Preprocessor
from .postprocessor import Postprocessor
from .config import (
    ConfigValidator,
    StringField,
    PathField,
    ListField,
    DictField,
    BaseField,
    NumberField,
    ConfigError,
    BoolField
)
from .dependency import UnregisteredProviderException
from .utils import (
    JSONDecoderWithAutoConversion,
    read_json, read_yaml,
    get_path, contains_all, set_image_metadata, OrderedSet, contains_any
)

from .representation import (
    BaseRepresentation, ReIdentificationClassificationAnnotation, ReIdentificationAnnotation, PlaceRecognitionAnnotation
)
from .data_readers import (
    DataReaderField, REQUIRES_ANNOTATIONS, BaseReader,
    serialize_identifier, deserialize_identifier, create_identifier_key
)
from .logging import print_info


class Dataset:
    def __init__(self, config_entry, delayed_annotation_loading=False):
        self.name = config_entry.get('name')
        self._config = config_entry
        self._batch = self.config.get('batch')
        self.iteration = 0
        self.data_provider = None
        ConfigValidator('dataset', fields=self.parameters()).validate(self.config)
        if not delayed_annotation_loading:
            self.create_data_provider()

    @classmethod
    def parameters(cls):
        return {
            'name': StringField(description='Dataset name'),
            'annotation': PathField(
                optional=True, check_exists=False, description='file for reading/writing Accuracy Checker annotation'
            ),
            'data_source': PathField(optional=True, check_exists=False, description='data source'),
            'dataset_meta': PathField(optional=True, check_exists=False, description='dataset metadata file'),
            'metrics': ListField(allow_empty=False, optional=True, description='list of metrics for evaluation'),
            'postprocessing': ListField(allow_empty=False, optional=True, description='list of postprocessings'),
            'preprocessing': ListField(allow_empty=False, optional=True, description='list of preprocessings'),
            'reader': DataReaderField(optional=True, description='data reader'),
            'annotation_conversion': DictField(optional=True, description='annotation conversion parameters'),
            'subsample_size': BaseField(optional=True, description='subset size for evaluation'),
            'shuffle': BoolField(optional=True, description='samples shuffling allowed or not'),
            'subsample_seed': NumberField(value_type=int, min_value=0, optional=True, description=''),
            'analyze_dataset': BoolField(optional=True, description='provide dataset analysis or not'),
            'segmentation_masks_source': PathField(
                is_directory=True, optional=True, description='additional data source for segmentation mask loading'
            ),
            'additional_data_source': PathField(
                is_directory=True, optional=True, description='additional data source for annotation loading'
            ),
            'subset_file': PathField(optional=True, description='file with identifiers for subset', check_exists=False),
            'store_subset': BoolField(
                optional=True, default=False,
                description='save subset ids to file specified in subset_file parameter'
            ),
            'batch': NumberField(value_type=int, min_value=1, optional=True, description='batch size for data read'),
            '_profile': BoolField(optional=True, default=False, description='allow metric profiling'),
            '_report_type': StringField(optional=True, choices=['json', 'csv'], description='type profiling report'),
            '_ie_preprocessing': BoolField(optional=True, default=False)
        }

    @staticmethod
    def load_annotation(config):
        def _convert_annotation():
            print_info("Annotation conversion for {dataset_name} dataset has been started".format(
                dataset_name=config['name']))
            print_info("Parameters to be used for conversion:")
            for key, value in config['annotation_conversion'].items():
                print_info('{key}: {value}'.format(key=key, value=value))
            annotation, meta = Dataset.convert_annotation(config)
            if annotation is not None:
                print_info("Annotation conversion for {dataset_name} dataset has been finished".format(
                    dataset_name=config['name']))
            return annotation, meta

        def _run_dataset_analysis(meta):
            if config.get('segmentation_masks_source'):
                meta['segmentation_masks_source'] = config.get('segmentation_masks_source')
            meta = analyze_dataset(annotation, meta)
            if meta.get('segmentation_masks_source'):
                del meta['segmentation_masks_source']
            return meta

        def _save_annotation():
            annotation_name = config['annotation']
            meta_name = config.get('dataset_meta')
            if meta_name:
                meta_name = Path(meta_name)
                print_info("{dataset_name} dataset metadata will be saved to {file}".format(
                    dataset_name=config['name'], file=meta_name))
            print_info('Converted annotation for {dataset_name} dataset will be saved to {file}'.format(
                dataset_name=config['name'], file=Path(annotation_name)))
            save_annotation(annotation, meta, Path(annotation_name), meta_name, config)

        annotation, meta = None, None
        use_converted_annotation = True
        if 'annotation' in config:
            annotation_file = Path(config['annotation'])
            if annotation_file.exists():
                print_info('Annotation for {dataset_name} dataset will be loaded from {file}'.format(
                    dataset_name=config['name'], file=annotation_file))
                annotation = read_annotation(get_path(annotation_file))
                meta = Dataset.load_meta(config)
                use_converted_annotation = False

        if not annotation and 'annotation_conversion' in config:
            annotation, meta = _convert_annotation()

        if not annotation:
            raise ConfigError('path to converted annotation or data for conversion should be specified')
        annotation = _create_subset(annotation, config)
        dataset_analysis = config.get('analyze_datase', False)

        if dataset_analysis:
            meta = _run_dataset_analysis(meta)

        if use_converted_annotation and contains_all(config, ['annotation', 'annotation_conversion']):
            _save_annotation()

        return annotation, meta

    def send_annotation_info(self, config):
        info = {
            'convert_annotation': False,
            'converter': None,
            'dataset_analysis': config.get('analyze_dataset', False),
            'annotation_saving': False,
            'dataset_size': self.size
        }
        subsample_size = config.get('subsample_size')
        subsample_meta = {'subset': False, 'shuffle': False}
        if not ignore_subset_settings(config):

            if subsample_size is not None:
                shuffle = config.get('shuffle', True)
                subsample_meta = {
                    'shuffle': shuffle,
                    'subset': True
                }

        info['subset_info'] = subsample_meta
        if 'annotation' in config:
            annotation_file = Path(config['annotation'])
            if annotation_file.exists():
                return info

        if 'annotation_conversion' in config:
            info['converter'] = config['annotation_conversion'].get('converter')

        if contains_all(config, ['annotation', 'annotation_conversion']):
            info['annotation_saving'] = True

        return info

    def create_data_provider(self):
        annotation, meta = self.load_annotation(self.config)
        data_reader_config = self.config.get('reader', 'opencv_imread')
        data_source = self.config.get('data_source')
        if isinstance(data_reader_config, str):
            data_reader_type = data_reader_config
            data_reader_config = None
        elif isinstance(data_reader_config, dict):
            data_reader_type = data_reader_config['type']
        else:
            raise ConfigError('reader should be dict or string')
        annotation_provider = AnnotationProvider(annotation, meta)
        if data_reader_type in REQUIRES_ANNOTATIONS:
            data_source = annotation_provider
        data_reader = BaseReader.provide(data_reader_type, data_source, data_reader_config)
        self.data_provider = DataProvider(
            data_reader, annotation_provider, dataset_config=self.config, batch=self.batch
        )

    @property
    def batch(self):
        return self._batch

    @batch.setter
    def batch(self, b):
        self._batch = b
        if self.data_provider:
            self.data_provider.batch = b

    @property
    def config(self):
        return deepcopy(self._config) #read-only

    def __len__(self):
        return len(self.data_provider)

    @property
    def size(self):
        return self.__len__()

    @property
    def full_size(self):
        return self.data_provider.full_size

    def __getitem__(self, item):
        return self.data_provider[item]

    @staticmethod
    def load_meta(config):
        meta = None
        meta_data_file = config.get('dataset_meta')
        if meta_data_file:
            print_info('{dataset_name} dataset metadata will be loaded from {file}'.format(
                dataset_name=config['name'], file=meta_data_file))
            meta = read_json(meta_data_file, cls=JSONDecoderWithAutoConversion)
        return meta

    @staticmethod
    def convert_annotation(config):
        conversion_params = config.get('annotation_conversion')
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
        self.data_provider.reset()
        if reload_annotation:
            self.data_provider.set_annotation(*self.load_annotation(self.config))

    def set_annotation(self, annotation, meta=None):
        if meta is None:
            meta = self.load_meta(self._config)
        self.data_provider.set_annotation(annotation, meta)

    def provide_data_info(self, annotations, progress_reporter=None):
        return self.data_provider.provide_data_info(annotations, progress_reporter)

    @classmethod
    def validate_config(cls, config, fetch_only=False, uri_prefix=''):
        dataset_config = ConfigValidator(
            uri_prefix or 'dataset', fields=cls.parameters(), on_extra_argument=ConfigValidator.ERROR_ON_EXTRA_ARGUMENT
        )
        errors = dataset_config.validate(config, fetch_only=fetch_only)
        if 'annotation_conversion' in config:
            conversion_uri = '{}.annotation_conversion'.format(uri_prefix) if uri_prefix else 'annotation_conversion'
            conversion_params = config['annotation_conversion']
            converter = conversion_params.get('converter')
            if not converter:
                error = ConfigError(
                    'converter is not found', conversion_params, conversion_uri, validation_scheme=BaseFormatConverter
                )
                if not fetch_only:
                    raise error
                errors.append(error)
                return errors
            try:
                converter_cls = BaseFormatConverter.resolve(converter)
            except UnregisteredProviderException as exception:
                if not fetch_only:
                    raise exception
                errors.append(
                    ConfigError(
                        'converter {} unregistered'.format(converter),
                        conversion_params, conversion_uri, validation_scheme=BaseFormatConverter)
                )
                return errors
            errors.extend(
                converter_cls.validate_config(conversion_params, fetch_only=fetch_only, uri_prefix=conversion_uri)
            )
        if not contains_any(config, ['annotation_conversion', 'annotation']):
            errors.append(
                ConfigError(
                    'annotation_conversion or annotation field should be provided', config, uri_prefix or 'dataset',
                    validation_scheme=cls.validation_scheme()
                )
            )
        return errors

    @classmethod
    def validation_scheme(cls):
        scheme = {param: field for param, field in cls.parameters().items() if not param.startswith('_')}
        scheme.update({
            'preprocessing': Preprocessor,
            'postprocessing': Postprocessor,
            'metrics': Metric,
            'reader': BaseReader,
            'annotation_conversion': BaseFormatConverter
        })
        return [scheme]

    @property
    def metadata(self):
        return self.data_provider.metadata

    @property
    def identifiers(self):
        return self.data_provider.identifiers

    @property
    def multi_infer(self):
        return self.data_provider.multi_infer

    @property
    def labels(self):
        return self.data_provider.labels


def read_annotation(annotation_file: Path):
    annotation_file = get_path(annotation_file)

    result = []
    with annotation_file.open('rb') as file:
        try:
            first_obj = pickle.load(file)
            if isinstance(first_obj, DatasetConversionInfo):
                describe_cached_dataset(first_obj)
            else:
                result.append(first_obj)
        except EOFError:
            return result
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


def describe_cached_dataset(dataset_info):
    print_info('Loaded dataset info:')
    if dataset_info.dataset_name:
        print_info('\tDataset name: {}'.format(dataset_info.dataset_name))
    print_info('\tAccuracy Checker version {}'.format(dataset_info.ac_version))
    print_info('\tDataset size {}'.format(dataset_info.dataset_size))
    print_info('\tConversion parameters:')
    for key, value in dataset_info.conversion_parameters.items():
        print_info('\t\t{key}: {value}'.format(key=key, value=value))
    if dataset_info.subset_parameters:
        print_info('\nSubset selection parameters:')
        for key, value in dataset_info.subset_parameters.items():
            print_info('\t\t{key}: {value}'.format(key=key, value=value))


class AnnotationProvider:
    def __init__(self, annotations, meta, name='', config=None):
        self.name = name
        self.config = config
        self._data_buffer = OrderedDict()
        self._meta = meta
        for ann in annotations:
            idx = create_identifier_key(ann.identifier)
            self._data_buffer[idx] = ann

    def __getitem__(self, item):
        return self._data_buffer[item]

    @property
    def identifiers(self):
        return list(self._data_buffer)

    def __len__(self):
        return len(self._data_buffer)

    def make_subset(self, ids=None, start=0, step=1, end=None, accept_pairs=False):
        pairwise_subset = isinstance(
            next(iter(self._data_buffer.values())), (
                ReIdentificationAnnotation,
                ReIdentificationClassificationAnnotation,
                PlaceRecognitionAnnotation
            )
        )
        if ids:
            return ids if not pairwise_subset else self._make_subset_pairwise(ids, accept_pairs)
        if not end:
            end = self.__len__()
        ids = range(start, end, step)
        return ids if not pairwise_subset else self._make_subset_pairwise(ids, accept_pairs)

    def _make_subset_pairwise(self, ids, add_pairs=False):
        def reid_pairwise_subset(pairs_set, subsample_set, ids):
            identifier_to_index = {
                idx: index for index, idx in enumerate(self._data_buffer)
            }
            index_to_identifier = dict(enumerate(self._data_buffer))
            for idx in ids:
                subsample_set.add(idx)
                current_annotation = self._data_buffer[index_to_identifier[idx]]
                positive_pairs = [
                    identifier_to_index[pair_identifier] for pair_identifier in current_annotation.positive_pairs
                ]
                pairs_set |= positive_pairs
                negative_pairs = [
                    identifier_to_index[pair_identifier] for pair_identifier in current_annotation.positive_pairs
                ]
                pairs_set |= negative_pairs
            return pairs_set, subsample_set

        def reid_subset(pairs_set, subsample_set, ids):
            index_to_identifier = dict(enumerate(self._data_buffer))
            for idx in ids:
                subsample_set.add(idx)
                selected_annotation = self._data_buffer[index_to_identifier[idx]]
                if not selected_annotation.query:
                    query_for_person = [
                        idx for idx, (_, annotation) in enumerate(self._data_buffer.items())
                        if annotation.person_id == selected_annotation.person_id and annotation.query
                    ]
                    pairs_set |= OrderedSet(query_for_person)
                else:
                    gallery_for_person = [
                        idx for idx, (_, annotation) in enumerate(self._data_buffer.items())
                        if annotation.person_id == selected_annotation.person_id and not annotation.query
                    ]
                    pairs_set |= OrderedSet(gallery_for_person)
            return pairs_set, subsample_set

        def ibl_subset(pairs_set, subsample_set, ids):
            queries_ids = [idx for idx, (_, ann) in enumerate(self._data_buffer.items()) if ann.query]
            gallery_ids = [idx for idx, (_, ann) in enumerate(self._data_buffer.items()) if not ann.query]
            subset_id_to_q_id = {s_id: idx for idx, s_id in enumerate(queries_ids)}
            subset_id_to_g_id = {s_id: idx for idx, s_id in enumerate(gallery_ids)}
            queries_loc = [ann.coords for ann in self._data_buffer.values() if ann.query]
            gallery_loc = [ann.coords for ann in self._data_buffer.values() if not ann.query]
            dist_mat = np.zeros((len(queries_ids), len(gallery_ids)))
            for idx, query_loc in enumerate(queries_loc):
                dist_mat[idx] = np.linalg.norm(np.array(query_loc) - np.array(gallery_loc), axis=1)
            for idx in ids:
                if idx in subset_id_to_q_id:
                    pair = gallery_ids[np.argmin(dist_mat[subset_id_to_q_id[idx]])]
                else:
                    pair = queries_ids[np.argmin(dist_mat[:, subset_id_to_g_id[idx]])]
                subsample_set.add(idx)
                pairs_set.add(pair)
            return pairs_set, subsample_set

        realisation = [
            (PlaceRecognitionAnnotation, ibl_subset),
            (ReIdentificationClassificationAnnotation, reid_pairwise_subset),
            (ReIdentificationAnnotation, reid_subset),
        ]
        subsample_set = OrderedSet()
        pairs_set = OrderedSet()
        for (dtype, func) in realisation:
            if isinstance(next(iter(self._data_buffer.values())), dtype):
                pairs_set, subsample_set = func(pairs_set, subsample_set, ids)
                break
        if add_pairs:
            subsample_set |= pairs_set

        return list(subsample_set)

    @property
    def metadata(self):
        return deepcopy(self._meta) #read-only

    @property
    def labels(self):
        return self._meta.get('label_map', {})


class DataProvider:
    def __init__(
            self, data_reader, annotation_provider=None, tag='', dataset_config=None, data_list=None, subset=None,
            batch=None
    ):
        self.tag = tag
        self.data_reader = data_reader
        self.annotation_provider = annotation_provider
        self.dataset_config = dataset_config or {}
        self.batch = batch if batch is not None else dataset_config.get('batch')
        self.subset = subset
        self.create_data_list(data_list)
        if self.store_subset:
            self.sava_subset()

    def create_data_list(self, data_list=None):
        if data_list is not None:
            self._data_list = data_list
            return
        self.store_subset = self.dataset_config.get('store_subset', False)

        if self.dataset_config.get('subset_file'):
            subset_file = Path(self.dataset_config['subset_file'])
            if subset_file.exists() and not self.store_subset:
                self.read_subset(subset_file)
                return

            self.store_subset = True

        if self.annotation_provider:
            self._data_list = self.annotation_provider.identifiers
            return
        self._data_list = [file.name for file in self.data_reader.data_source.glob('*')]

    def read_subset(self, subset_file):
        identifiers = [deserialize_identifier(idx) for idx in read_yaml(subset_file)]
        self._data_list = identifiers
        print_info("loaded {} data items from {}".format(len(self._data_list), subset_file))

    def sava_subset(self):
        identifiers = [serialize_identifier(idx) for idx in self._data_list]
        subset_file = Path(self.dataset_config.get(
            'subset_file', '{}_subset_{}.yml'.format(self.dataset_config['name'], len(identifiers))))
        print_info("Data subset will be saved to {} file".format(subset_file))
        with subset_file.open(mode="w") as sf:
            yaml.safe_dump(identifiers, sf)

    def __getitem__(self, item):
        if self.batch is None:
            self.batch = 1
        if self.size <= item * self.batch:
            raise IndexError
        batch_annotation = []
        batch_start = item * self.batch
        batch_end = min(self.size, batch_start + self.batch)
        batch_input_ids = self.subset[batch_start:batch_end] if self.subset else range(batch_start, batch_end)
        batch_identifiers = [self._data_list[idx] for idx in batch_input_ids]
        batch_input = [self.data_reader(identifier=identifier) for identifier in batch_identifiers]
        if self.annotation_provider:
            batch_annotation = [self.annotation_provider[idx] for idx in batch_identifiers]

            for annotation, input_data in zip(batch_annotation, batch_input):
                self.set_annotation_metadata(annotation, input_data, self.data_reader.data_source)

        return batch_input_ids, batch_annotation, batch_input, batch_identifiers

    def __len__(self):
        if self.subset is None:
            return len(self._data_list)
        return len(self.subset)

    @property
    def identifiers(self):
        return self._data_list

    def make_subset(self, ids=None, start=0, step=1, end=None, accept_pairs=False):
        if self.annotation_provider:
            ids = self.annotation_provider.make_subset(ids, start, step, end, accept_pairs)
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
        self._batch = batch

    @property
    def labels(self):
        return self.annotation_provider.metadata.get('label_map', {})

    @property
    def metadata(self):
        if self.annotation_provider:
            return self.annotation_provider.metadata
        return {}

    def reset(self, reload_annotation=False):
        if self.subset:
            self.subset = None
        if self.annotation_provider and reload_annotation:
            self.set_annotation(*Dataset.load_annotation(self.dataset_config))
        self.data_reader.reset()

    @property
    def full_size(self):
        if not self.annotation_provider:
            return len(self._data_list)
        return len(self.annotation_provider)

    @property
    def size(self):
        return self.__len__()

    @property
    def multi_infer(self):
        return getattr(self.data_reader, 'multi_infer', False)

    def set_annotation_metadata(self, annotation, image, data_source):
        set_image_metadata(annotation, image)
        annotation.set_data_source(data_source if not isinstance(data_source, (list, AnnotationProvider)) else [])
        segmentation_mask_source = self.dataset_config.get('segmentation_masks_source')
        annotation.set_segmentation_mask_source(segmentation_mask_source)
        annotation.set_additional_data_source(self.dataset_config.get('additional_data_source'))
        annotation.set_dataset_metadata(self.annotation_provider.metadata)

    def provide_data_info(self, annotations, progress_reporter=None):
        if progress_reporter:
            progress_reporter.reset(len(annotations))
        for idx, ann in enumerate(annotations):
            input_data = self.data_reader(ann.identifier)
            self.set_annotation_metadata(ann, input_data, self.data_reader.data_source)
            if progress_reporter:
                progress_reporter.update(idx, 1)
        return annotations

    def set_annotation(self, annotation, meta):
        subsample_size = self.dataset_config.get('subsample_size')
        if subsample_size is not None:
            subsample_seed = self.dataset_config.get('subsample_seed', 666)

            annotation = create_subset(annotation, subsample_size, subsample_seed)

        if self.dataset_config.get('analyze_dataset', False):
            if self.dataset_config.get('segmentation_masks_source'):
                meta['segmentation_masks_source'] = self.dataset_config.get('segmentation_masks_source')
            meta = analyze_dataset(annotation, meta)
            if meta.get('segmentation_masks_source'):
                del meta['segmentation_masks_source']
        self.annotation_provider = AnnotationProvider(annotation, meta)
        self.create_data_list()


class DatasetWrapper(DataProvider):
    pass


def ignore_subset_settings(config):
    subset_file = config.get('subset_file')
    store_subset = config.get('store_subset')
    if subset_file is None:
        return False
    if Path(subset_file).exists() and not store_subset:
        return True
    return False


def _create_subset(annotation, config):
    subsample_size = config.get('subsample_size')
    if not ignore_subset_settings(config):

        if subsample_size is not None:
            subsample_seed = config.get('subsample_seed', 666)
            shuffle = config.get('shuffle', True)
            annotation = create_subset(annotation, subsample_size, subsample_seed, shuffle)

    elif subsample_size is not None:
        warnings.warn("Subset selection parameters will be ignored")
        config.pop('subsample_size', None)
        config.pop('subsample_seed', None)
        config.pop('shuffle', None)

    return annotation
