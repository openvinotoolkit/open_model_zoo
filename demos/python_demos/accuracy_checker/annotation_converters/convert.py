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

import json
from pathlib import Path
from argparse import ArgumentParser
from functools import partial

import numpy as np

from accuracy_checker.utils import get_path

from .format_converter import BaseFormatConverter


def build_argparser():
    parser = ArgumentParser(
        description="Converts annotation form a arbitrary format to accuracy-checker specific format", add_help=False
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

    return parser


def make_subset(annotation, size):
    np.random.seed(666)
    return list(np.random.choice(annotation, size=size, replace=False))


def main():
    main_argparser = build_argparser()
    args, _ = main_argparser.parse_known_args()

    converter = BaseFormatConverter.provide(args.converter)
    converter_argparser = converter.get_argparser()
    converter_options, _ = converter_argparser.parse_known_args()

    main_argparser = ArgumentParser(parents=[main_argparser, converter_argparser])
    args = main_argparser.parse_args()

    converter_arguments = {k: v for k, v in vars(args).items() if k in vars(converter_options)}

    out_dir = args.output_dir or Path.cwd()
    annotation_name = args.annotation_name or "{}.pickle".format(converter.get_name())
    meta_name = args.meta_name or "{}.json".format(converter.get_name())

    output_file = out_dir / annotation_name
    output_meta = out_dir / meta_name

    result, meta = converter.convert(**converter_arguments)

    subsample = args.subsample
    if subsample:
        if subsample.endswith('%'):
            subsample_ratio = float(subsample[:-1]) / 100
            subsample_size = int(len(result) * subsample_ratio)
        else:
            subsample_size = int(subsample)

        result = make_subset(result, subsample_size)

    save_annotation(result, meta, output_file, output_meta)


def save_annotation(annotation, meta, output_file, output_meta):
    with output_file.open('wb') as file:
        for representation in annotation:
            representation.dump(file)

    if meta:
        with output_meta.open('wt') as file:
            json.dump(meta, file)


if __name__ == '__main__':
    main()
