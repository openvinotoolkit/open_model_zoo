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
import os
import pathlib
from argparse import ArgumentParser
from collections import Counter

import cv2
import numpy as np
from tqdm import tqdm

from accuracy_checker.dataset import Dataset
from accuracy_checker.logging import print_info
from accuracy_checker.representation import ClassificationAnnotation, DetectionAnnotation

from .format_converter import BaseFormatConverter

registered_converters = list(BaseFormatConverter.providers.keys())

cwd = pathlib.Path.cwd()


def path_arg(check=False, is_dir=False):
    def get_arg(arg):
        path = pathlib.Path(arg).absolute()

        if check and not path.exists():
            raise FileNotFoundError
        if check and is_dir and not path.is_dir():
            raise FileNotFoundError("expected directory")

        return path

    return get_arg


def build_argparser():
    parser = ArgumentParser(
        description="Converts annotation form a arbitrary format to accuracy-checker specific format", add_help=False)
    parser.add_argument("converter", help="Specific converter to run", choices=registered_converters)
    parser.add_argument("-o", "--output_dir", help="Directory to save converted annotation and meta info",
                        required=False,
                        type=path_arg(check=True, is_dir=True))
    parser.add_argument("-m", "--meta_name", help="Meta info file name", required=False)
    parser.add_argument("-a", "--annotation_name", help="Annotation file name", required=False)
    parser.add_argument("-d", "--data_dir", help="Directory with source data", required=False,
                        type=path_arg(check=True, is_dir=True))
    parser.add_argument("-ss", "--subsample", help="Dataset subsample size", required=False)
    parser.add_argument("-sx", "--stratified_subset", help="Make subsample with equal label distribution",
                        required=False,
                        action="store_true")
    parser.add_argument("-im", "--image_meta", help="Read images and save image level metadata", action="store_true",
                        required=False)

    return parser


def make_subset(annotation, size, number_labels, stratified=False):
    np.random.seed(666)
    if not stratified:
        return list(np.random.choice(annotation, size=size, replace=False))
    objects_per_label = size // number_labels

    np.random.shuffle(annotation)
    if isinstance(annotation[0], ClassificationAnnotation):
        return _classification_stratified_subset(annotation, objects_per_label, size)

    if isinstance(annotation[0], DetectionAnnotation):
        return _detection_stratified_subset(annotation, objects_per_label, size)

    raise ValueError("Unknown annotation type for converted dataset. Can not create stratified subsample")


def _detection_stratified_subset(annotation, objects_per_label, size):
    subsample = []
    for ann in annotation:
        counter = Counter(ann.labels)
        if len(subsample) >= size:
            break

        for count in counter.values():
            if count < objects_per_label:
                subsample.append(ann)
                for key in counter:
                    counter[key] += 1
    return subsample


def _classification_stratified_subset(annotation, objects_per_label, size):
    counter = Counter()
    subsample = []
    for ann in annotation:
        if len(subsample) >= size:
            break

        if counter[ann.label] < objects_per_label:
            subsample.append(ann)
            counter[ann.label] += 1
    return subsample


def load_image_meta(annotation, image_root):
    print_info("Loading image level metadata")
    for ann in tqdm(annotation):
        path = os.path.join(image_root, ann.identifier)
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError('Image file not found: {}'.format(path))
        Dataset.set_image_metadata(ann, image)
    return annotation


def main():
    main_argparser = build_argparser()
    args, _ = main_argparser.parse_known_args()

    converter = BaseFormatConverter.provide(args.converter)
    converter_argparser = converter.get_argparser()
    converter_options, _ = converter_argparser.parse_known_args()

    main_argparser = ArgumentParser(parents=[main_argparser, converter_argparser])
    args = main_argparser.parse_args()

    converter_arguments = {k: v for k, v in vars(args).items() if k in vars(converter_options)}

    out_dir = args.output_dir or cwd
    annotation_name = args.annotation_name or "{}.pickle".format(converter.get_name())
    meta_name = args.meta_name or "{}.json".format(converter.get_name())

    output_file = out_dir / annotation_name
    output_meta = out_dir / meta_name

    result, meta = converter.convert(**converter_arguments)

    stratified = args.stratified_subset
    subsample = args.subsample
    if subsample:
        label_map = meta.get('label_map') if meta else None
        if not label_map:
            raise ValueError('subsample mode is supported only if label_map is provided')

        if subsample.endswith('%'):
            subsample_ratio = float(subsample[:-1]) / 100
            subsample_size = int(len(result) * subsample_ratio)
        else:
            subsample_size = int(subsample)

        result = make_subset(result, subsample_size, len(label_map), stratified)

    if args.image_meta:
        data_dir = args.data_dir.as_posix() if args.data_dir else None
        image_root = data_dir or converter.image_root
        if not image_root:
            raise ValueError(
                "Either `data_dir` command line option must be provided "
                "or annotation converter must set `image_root` field."
            )
        result = load_image_meta(result, image_root)

    save_annotation(result, meta, output_file, output_meta)


def save_annotation(annotation, meta, output_file, output_meta):
    with output_file.open('wb') as file:
        for representation in annotation:
            representation.dump(file)
    if meta is not None:
        with output_meta.open('wt') as file:
            json.dump(meta, file)


if __name__ == '__main__':
    main()
