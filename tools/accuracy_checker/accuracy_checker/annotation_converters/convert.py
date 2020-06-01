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

import warnings

import copy
import json
from pathlib import Path
from argparse import ArgumentParser
from functools import partial

import numpy as np

from ..representation import ReIdentificationClassificationAnnotation
from ..utils import get_path, OrderedSet
from ..data_analyzer import BaseDataAnalyzer
from .format_converter import BaseFormatConverter
from ..utils import cast_to_bool


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

        for ann_ind in potential_ann_ind: # pylint: disable=E1133
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

    np.random.seed(seed)
    dataset_size = len(annotation)
    if dataset_size < size:
        warnings.warn('Dataset size {} less than subset size {}'.format(dataset_size, size))
        return annotation
    if isinstance(annotation[-1], ReIdentificationClassificationAnnotation):
        return make_subset_pairwise(annotation, size, shuffle)

    result_annotation = list(np.random.choice(annotation, size=size, replace=False)) if shuffle else annotation[:size]
    return result_annotation


def main():
    main_argparser = build_argparser()
    args, _ = main_argparser.parse_known_args()
    converter, converter_argparser, converter_args = get_converter_arguments(args)

    main_argparser = ArgumentParser(parents=[main_argparser, converter_argparser])
    args = main_argparser.parse_args()

    converter = configure_converter(converter_args, args, converter)
    out_dir = args.output_dir or Path.cwd()

    results = converter.convert()
    converted_annotation = results.annotations
    meta = results.meta
    errors = results.content_check_errors
    if errors:
        warnings.warn('Following problems were found during conversion:'
                      '\n{}'.format('\n'.join(errors)))

    subsample = args.subsample
    if subsample:
        if subsample.endswith('%'):
            subsample_ratio = float(subsample[:-1]) / 100
            subsample_size = int(len(converted_annotation) * subsample_ratio)
        else:
            subsample_size = int(args.subsample)

        converted_annotation = make_subset(converted_annotation, subsample_size, args.subsample_seed, args.shuffle)

    if args.analyze_dataset:
        analyze_dataset(converted_annotation, meta)

    converter_name = converter.get_name()
    annotation_name = args.annotation_name or "{}.pickle".format(converter_name)
    meta_name = args.meta_name or "{}.json".format(converter_name)

    annotation_file = out_dir / annotation_name
    meta_file = out_dir / meta_name

    save_annotation(converted_annotation, meta, annotation_file, meta_file)


def save_annotation(annotation, meta, annotation_file, meta_file):
    if annotation_file:
        annotation_dir = annotation_file.parent
        if not annotation_dir.exists():
            annotation_dir.mkdir(parents=True)
        with annotation_file.open('wb') as file:
            for representation in annotation:
                representation.dump(file)
    if meta_file and meta:
        meta_dir = meta_file.parent
        if not meta_dir.exists():
            meta_dir.mkdir(parents=True)
        with meta_file.open('wt') as file:
            json.dump(meta, file)


def configure_converter(converter_options, args, converter):
    args_dict, converter_options_dict = vars(args), vars(converter_options)
    converter_config = {
        option_name: option_value for option_name, option_value in args_dict.items()
        if option_name in converter_options_dict and option_value is not None
    }
    converter_config['converter'] = args.converter
    converter.config = converter_config
    converter.validate_config()
    converter.configure()

    return converter


def get_converter_arguments(arguments):
    converter = BaseFormatConverter.provide(arguments.converter)
    converter_argparser = converter.get_argparser()
    converter_options, _ = converter_argparser.parse_known_args()
    return converter, converter_argparser, converter_options


def analyze_dataset(annotations, metadata):
    first_element = next(iter(annotations), None)
    analyzer = BaseDataAnalyzer.provide(first_element.__class__.__name__)
    inside_meta = copy.copy(metadata)
    analyzer.analyze(annotations, inside_meta)
