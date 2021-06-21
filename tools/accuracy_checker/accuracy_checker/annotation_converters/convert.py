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

import warnings
import platform

import copy
import json
from pathlib import Path
import pickle
from argparse import ArgumentParser
from collections import namedtuple
from functools import partial

import numpy as np

from .. import __version__
from ..representation import (
    ReIdentificationClassificationAnnotation, ReIdentificationAnnotation, PlaceRecognitionAnnotation,
)
from ..data_readers import KaldiFrameIdentifier, KaldiMatrixIdentifier
from ..utils import (
    get_path, OrderedSet, cast_to_bool, is_relative_to, start_telemetry, send_telemetry_event, end_telemetry
)
from ..data_analyzer import BaseDataAnalyzer
from .format_converter import BaseFormatConverter

DatasetConversionInfo = namedtuple('DatasetConversionInfo',
                                   [
                                       'dataset_name',
                                       'conversion_parameters',
                                       'subset_parameters',
                                       'dataset_size',
                                       'ac_version'
                                   ])


def build_argparser():
    parser = ArgumentParser(
        description="Converts annotation form an arbitrary format to accuracy-checker specific format", add_help=False
    )
    parser.add_argument(
        "converter",
        help="Specific converter to run",
        choices=list(BaseFormatConverter.providers.keys())
    )
    parser.add_argument(
        "-o", "--output_dir",
        help="Directory to save converted annotation and meta info",
        required=False,
        type=partial(get_path, is_directory=True)
    )
    parser.add_argument("-m", "--meta_name", help="Meta info file name", required=False)
    parser.add_argument("-a", "--annotation_name", help="Annotation file name", required=False)
    parser.add_argument("-ss", "--subsample", help="Dataset subsample size", required=False)
    parser.add_argument(
        "--subsample_seed", help="Seed for generation dataset subsample", type=int, required=False, default=666
    )
    parser.add_argument('--analyze_dataset', required=False, action='store_true')
    parser.add_argument(
        "--shuffle",
        help="Allow shuffle annotation during creation a subset",
        required=False,
        type=cast_to_bool,
        default=True
    )

    return parser


def make_subset(annotation, size, seed=666, shuffle=True):
    np.random.seed(seed)
    dataset_size = len(annotation)
    if dataset_size < size:
        warnings.warn('Dataset size {} less than subset size {}'.format(dataset_size, size))
        return annotation
    if isinstance(annotation[-1], ReIdentificationClassificationAnnotation):
        return make_subset_pairwise(annotation, size, shuffle)
    if isinstance(annotation[-1], ReIdentificationAnnotation):
        return make_subset_reid(annotation, size, shuffle)
    if isinstance(annotation[-1], PlaceRecognitionAnnotation):
        return make_subset_place_recognition(annotation, size, shuffle)
    if isinstance(annotation[-1].identifier, (KaldiMatrixIdentifier, KaldiFrameIdentifier)):
        return make_subset_kaldi(annotation, size, shuffle)

    result_annotation = list(np.random.choice(annotation, size=size, replace=False)) if shuffle else annotation[:size]
    return result_annotation


def make_subset_pairwise(annotation, size, shuffle=True):
    def get_pairs(pairs_list):
        pairs_set = OrderedSet()
        for identifier in pairs_list:
            next_annotation = next(
                pair_annotation for pair_annotation in annotation if pair_annotation.identifier == identifier
            )
            positive_pairs = get_pairs(next_annotation.positive_pairs)
            negative_pairs = get_pairs(next_annotation.negative_pairs)
            pairs_set.add(next_annotation)
            pairs_set |= positive_pairs
            pairs_set |= negative_pairs

        return pairs_set

    subsample_set = OrderedSet()

    potential_ann_ind = np.random.choice(len(annotation), size, replace=False) if shuffle else np.arange(size)

    for ann_ind in potential_ann_ind:  # pylint: disable=E1133
        annotation_for_subset = annotation[ann_ind]
        positive_pairs = annotation_for_subset.positive_pairs
        negative_pairs = annotation_for_subset.negative_pairs
        if len(positive_pairs) + len(negative_pairs) == 0:
            continue
        updated_pairs = OrderedSet()
        updated_pairs.add(annotation_for_subset)
        updated_pairs |= get_pairs(positive_pairs)
        updated_pairs |= get_pairs(negative_pairs)
        intersection = subsample_set & updated_pairs
        subsample_set |= updated_pairs
        if len(subsample_set) == size:
            break
        if len(subsample_set) > size:
            to_delete = updated_pairs - intersection
            subsample_set -= to_delete

    return list(subsample_set)


def make_subset_reid(annotation, size, shuffle=True):
    subsample_set = OrderedSet()
    potential_ann_ind = np.random.choice(len(annotation), size, replace=False) if shuffle else np.arange(size)
    for ann_ind in potential_ann_ind:
        selected_annotation = annotation[ann_ind]
        if not selected_annotation.query:
            query_for_person = [
                ann for ann in annotation if ann.person_id == selected_annotation.person_id and ann.query
            ]
            pairs_set = OrderedSet(query_for_person)
        else:
            gallery_for_person = [
                ann for ann in annotation
                if ann.person_id == selected_annotation.person_id and not ann.query
            ]
            pairs_set = OrderedSet(gallery_for_person)
        if len(pairs_set) == 0:
            continue
        subsample_set.add(selected_annotation)
        intersection = subsample_set & pairs_set
        subsample_set |= pairs_set
        if len(subsample_set) == size:
            break
        if len(subsample_set) > size:
            pairs_set.add(selected_annotation)
            to_delete = pairs_set - intersection
            subsample_set -= to_delete

    return list(subsample_set)


def make_subset_place_recognition(annotation, size, shuffle=True):
    subsample_set = OrderedSet()
    potential_ann_ind = np.random.choice(len(annotation), size, replace=False) if shuffle else np.arange(size)
    queries_ids = [idx for idx, ann in enumerate(annotation) if ann.query]
    gallery_ids = [idx for idx, ann in enumerate(annotation) if not ann.query]
    subset_id_to_q_id = {s_id: idx for idx, s_id in enumerate(queries_ids)}
    subset_id_to_g_id = {s_id: idx for idx, s_id in enumerate(gallery_ids)}
    queries_loc = [ann.coords for ann in annotation if ann.query]
    gallery_loc = [ann.coords for ann in annotation if not ann.query]
    dist_mat = np.zeros((len(queries_ids), len(gallery_ids)))
    for idx, query_loc in enumerate(queries_loc):
        dist_mat[idx] = np.linalg.norm(np.array(query_loc) - np.array(gallery_loc), axis=1)
    for idx in potential_ann_ind:
        if idx in subset_id_to_q_id:
            pair = gallery_ids[np.argmin(dist_mat[subset_id_to_q_id[idx]])]
        else:
            pair = queries_ids[np.argmin(dist_mat[:, subset_id_to_g_id[idx]])]
        addition = OrderedSet([idx, pair])
        subsample_set |= addition
        if len(subsample_set) == size:
            break
        if len(subsample_set) > size:
            subsample_set -= addition
    return [annotation[ind] for ind in subsample_set]


def make_subset_kaldi(annotation, size, shuffle=True):
    file_to_num_utterances = {}
    for ind, ann in enumerate(annotation):
        if ann.identifier.file not in file_to_num_utterances:
            file_to_num_utterances[ann.identifier.file] = []
        file_to_num_utterances[ann.identifier.file].append(ind)

    subset = []
    for _, indices in file_to_num_utterances.items():
        if len(subset) + len(indices) > size:
            num_elem_to_add = size - len(subset)
            if shuffle:
                indices = np.random.choice(indices, num_elem_to_add)
            else:
                indices = indices[:num_elem_to_add]
        else:
            if shuffle:
                indices = np.random.shuffle(indices)
        subset.extend([annotation[idx] for idx in indices])
        if len(subset) == size:
            break
    return subset


def main():
    main_argparser = build_argparser()
    tm = start_telemetry()

    args, _ = main_argparser.parse_known_args()
    converter, converter_argparser, converter_args = get_converter_arguments(args)
    details = {
        'platform': platform.system(), 'conversion_errors': None, 'save_annotation': True,
        'subsample': bool(args.subsample),
        'shuffle': args.shuffle,
        'converter': converter.get_name(),
        'dataset_analysis': args.analyze_dataset
    }

    main_argparser = ArgumentParser(parents=[main_argparser, converter_argparser])
    args = main_argparser.parse_args()

    converter, converter_config = configure_converter(converter_args, args, converter)
    out_dir = args.output_dir or Path.cwd()

    results = converter.convert()
    converted_annotation = results.annotations
    meta = results.meta
    errors = results.content_check_errors
    if errors:
        warnings.warn('Following problems were found during conversion:'
                      '\n{}'.format('\n'.join(errors)))
        details['conversion_errors'] = str(len(errors))

    subsample = args.subsample
    if subsample:
        if subsample.endswith('%'):
            subsample_ratio = float(subsample[:-1]) / 100
            subsample_size = int(len(converted_annotation) * subsample_ratio)
        else:
            subsample_size = int(args.subsample)

        converted_annotation = make_subset(converted_annotation, subsample_size, args.subsample_seed, args.shuffle)
        details['dataset_size'] = len(converted_annotation)
    send_telemetry_event(tm, 'annotation_conversion', json.dumps(details))
    if args.analyze_dataset:
        analyze_dataset(converted_annotation, meta)

    converter_name = converter.get_name()
    annotation_name = args.annotation_name or "{}.pickle".format(converter_name)
    meta_name = args.meta_name or "{}.json".format(converter_name)

    annotation_file = out_dir / annotation_name
    meta_file = out_dir / meta_name
    dataset_config = {
        'name': annotation_name,
        'annotation_conversion': converter_config,
    }

    save_annotation(converted_annotation, meta, annotation_file, meta_file, dataset_config)
    end_telemetry(tm)


def save_annotation(annotation, meta, annotation_file, meta_file, dataset_config=None):
    if annotation_file:
        conversion_meta = get_conversion_attributes(dataset_config, len(annotation)) if dataset_config else None
        annotation_dir = annotation_file.parent
        if not annotation_dir.exists():
            annotation_dir.mkdir(parents=True)
        with annotation_file.open('wb') as file:
            if conversion_meta:
                pickle.dump(conversion_meta, file)
            for representation in annotation:
                representation.dump(file)

    if meta_file and meta:
        meta_dir = meta_file.parent
        if not meta_dir.exists():
            meta_dir.mkdir(parents=True)
        with meta_file.open('wt') as file:
            json.dump(meta, file)


def get_conversion_attributes(config, dataset_size):
    dataset_name = config.get('name', '')
    conversion_parameters = copy.deepcopy(config.get('annotation_conversion', {}))
    for key, value in config.get('annotation_conversion', {}).items():
        if key in config.get('_command_line_mapping', {}):
            m_path = config['_command_line_mapping'][key]
            if not m_path:
                conversion_parameters[key] = str(value)
                continue

            if isinstance(m_path, list):
                path_list = []
                for path in m_path:
                    if isinstance(path, list):
                        path_list.extend(path)
                    else:
                        path_list.append(path)

                for m_path in path_list:
                    if is_relative_to(value, m_path):
                        break
            conversion_parameters[key] = str(value.relative_to(m_path))
        if isinstance(value, Path):
            conversion_parameters[key] = 'PATH/{}'.format(value.name)

    subset_size = config.get('subsample_size')
    subset_parameters = {}
    if subset_size is not None:
        shuffle = config.get('shuffle', True)
        subset_parameters = {
            'subsample_size': subset_size,
            'subsample_seed': config.get('subsample_seed'),
            'shuffle': shuffle
        }
    return DatasetConversionInfo(dataset_name, conversion_parameters, subset_parameters, dataset_size, __version__)


def configure_converter(converter_options, args, converter):
    args_dict, converter_options_dict = vars(args), vars(converter_options)
    converter_config = {
        option_name: option_value for option_name, option_value in args_dict.items()
        if option_name in converter_options_dict and option_value is not None
    }
    converter_config['converter'] = args.converter
    converter.config = converter_config
    converter.validate_config(converter_config)
    converter.configure()

    return converter, converter_config


def get_converter_arguments(arguments):
    converter = BaseFormatConverter.provide(arguments.converter, {})
    converter_argparser = converter.get_argparser()
    converter_options, _ = converter_argparser.parse_known_args()
    return converter, converter_argparser, converter_options


def analyze_dataset(annotations, metadata):
    first_element = next(iter(annotations), None)
    analyzer = BaseDataAnalyzer.provide(first_element.__class__.__name__)
    inside_meta = copy.copy(metadata)
    data_analysis = analyzer.analyze(annotations, inside_meta)
    if metadata:
        metadata['data_analysis'] = data_analysis
    else:
        metadata = {'data_analysis': data_analysis}
    return metadata
