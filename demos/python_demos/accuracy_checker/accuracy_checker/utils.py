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

import collections
import errno
import itertools
import json
import os
import pickle
import csv
from pathlib import Path
from typing import Union

import numpy as np
import yaml

from .representation import BaseRepresentation

try:
    import lxml.etree as ET
except ImportError:
    import xml.etree.cElementTree as ET


def concat_lists(*lists):
    return list(itertools.chain(*lists))


def get_path(entry: Union[str, Path], is_directory=False):
    path = Path(entry)
    # pathlib.Path.exists throws an exception in case of broken symlink
    if not os.path.exists(str(path)):
        raise FileNotFoundError('{}: {}'.format(os.strerror(errno.ENOENT), path))

    if is_directory and not path.is_dir():
        raise NotADirectoryError('{}: {}'.format(os.strerror(errno.ENOTDIR), path))

    # if it exists it is either file (or valid symlink to file) or directory (or valid symlink to directory)
    if not is_directory and not path.is_file():
        raise IsADirectoryError('{}: {}'.format(os.strerror(errno.EISDIR), path))

    return path


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


def contains_all(container, *args):
    s = set(container)

    for arg in args:
        if len(s.intersection(arg)) != len(arg):
            return False

    return True


def contains_any(container, *args):
    s = set(container)

    for arg in args:
        if s.intersection(arg):
            return True

    return False


def string_to_tuple(string, casting_type=float):
    processed = string.replace(' ', '')
    processed = processed.replace('(', '')
    processed = processed.replace(')', '')
    processed = processed.split(',')

    return tuple([casting_type(x) for x in processed])


def string_to_list(string):
    processed = string.replace(' ', '')
    processed = processed.replace('[', '')
    processed = processed.replace(']', '')
    processed = processed.split(',')

    return list(x for x in processed)


class JSONDecoderWithAutoConversion(json.JSONDecoder):
    """ Custom json decoder to convert all strings into numbers (int, float) during reading json file """

    def decode(self, s, _w=json.decoder.WHITESPACE.match):
        decoded = super().decode(s, _w)
        return self._decode(decoded)

    def _decode(self, entry):
        if isinstance(entry, str):
            try:
                return int(entry)
            except ValueError:
                pass
            try:
                return float(entry)
            except ValueError:
                pass
        elif isinstance(entry, dict):
            return {self._decode(key): self._decode(value) for key, value in entry.items()}
        elif isinstance(entry, list):
            return [self._decode(value) for value in entry]

        return entry


def dict_subset(dict_, key_subset):
    return {k: v for k, v in dict_.items() if k in key_subset}


def zipped_transform(fn, *iterables, inplace=False):
    ret = (
        iterables if inplace
        else tuple([] for _ in range(len(iterables)))
    )
    upd_function = (
        list.__setitem__ if inplace
        else lambda lst, _, result: lst.append(result)
    )
    for i, it in enumerate(zip(*iterables)):
        iter_res = fn(*it)
        if not iter_res:
            continue
        for dst, res in zip(ret, iter_res):
            upd_function(dst, i, res)

    return ret


def overrides(obj, attribute_name, base=None):
    cls = obj if isinstance(obj, type) else obj.__class__

    base = cls.__bases__[0] if not base else base
    obj_attr = getattr(cls, attribute_name, None)
    base_attr = getattr(base, attribute_name, None)

    return obj_attr and obj_attr != base_attr


def enum_values(enum):
    return [member.value for member in enum]


def get_size_from_config(config, allow_none=False):
    if 'size' in config:
        return config['size'], config['size']
    if contains_all(config, ('dst_width', 'dst_height')):
        return config['dst_height'], config['dst_width']
    if not allow_none:
        raise ValueError('Either size or dst_width and dst_height required')

    return None, None


def in_interval(value, interval):
    min_ = interval[0]
    max_ = interval[1] if len(interval) >= 2 else None

    if max_ is None:
        return min_ <= value

    return min_ <= value < max_


def parse_inputs(inputs_entry):
    inputs = {}
    for input_ in inputs_entry:
        value = input_['value']
        if isinstance(value, list):
            value = np.array(value)

        inputs[input_['name']] = value

    return inputs


def check_user_inputs(network_inputs, user_inputs):
    for name in user_inputs:
        if name in network_inputs:
            continue

        raise ValueError('network does not contain input "{}"'.format(name))


def reshape_user_inputs(config_inputs, executable_inputs):
    for input_layer, _ in config_inputs.items():
        config_inputs[input_layer] = config_inputs[input_layer].reshape(executable_inputs[input_layer].shape)


def finalize_metric_result(values, names):
    result_values, result_names = [], []
    for value, name in zip(values, names):
        if np.isnan(value):
            continue
        result_values.append(value)
        result_names.append(name)

    return result_values, result_names


def get_representations(values, representation_source):
    return np.reshape([value.get(representation_source) for value in values], -1)


def get_supported_representations(container, supported_types):
    if np.shape(container) == ():
        container = [container]

    return list(filter(lambda rep: check_representation_type(rep, supported_types), container))


def check_representation_type(representation, representation_types):
    return isinstance(representation, representation_types)


def is_single_metric_source(source):
    if source is None:
        return False
    entries = source.split(',')

    return np.size(entries) == 1


def read_txt(file: Union[str, Path], sep='\n', **kwargs):
    def is_empty(string):
        return not string or string.isspace()

    with get_path(file).open() as content:
        content = content.read(**kwargs).split(sep)
        content = list(filter(lambda string: not is_empty(string), content))

        return list(map(str.strip, content))


def read_xml(file: Union[str, Path], *args, **kwargs):
    return ET.parse(str(get_path(file)), *args, **kwargs).getroot()


def read_json(file: Union[str, Path], *args, **kwargs):
    with get_path(file).open() as content:
        return json.load(content, *args, **kwargs)


def read_pickle(file: Union[str, Path], *args, **kwargs):
    with get_path(file).open('rb') as content:
        return pickle.load(content, *args, **kwargs)


def read_yaml(file: Union[str, Path], *args, **kwargs):
    # yaml does not keep order of keys in dictionaries but it is important for reading pre/post processing
    yaml.add_representer(collections.OrderedDict, lambda dumper, data: dumper.represent_dict(data.iteritems()))
    yaml.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        lambda loader, node: collections.OrderedDict(loader.construct_pairs(node))
    )

    with get_path(file).open() as content:
        return yaml.load(content, *args, **kwargs)


def read_csv(file: Union[str, Path], *args, **kwargs):
    with get_path(file).open() as content:
        reader = csv.DictReader(content, *args, **kwargs)
        return list(reader)


def extract_image_representations(image_representations):
    images = [rep.data for rep in image_representations]
    meta = [rep.metadata for rep in image_representations]
    return images, meta


def convert_bboxes_xywh_to_x1y1x2y2(x_coord, y_coord, width, height):
    return x_coord, y_coord, x_coord + width, y_coord + height


def get_or_parse_value(item, supported_values, default=None):
    if isinstance(item, str):
        item = item.lower()
        if item in supported_values:
            return supported_values[item]

        try:
            return string_to_tuple(item)
        except ValueError:
            message = 'Invalid value "{}", expected one of precomputed: ({}) or list of values'.format(
                item, ', '.join(supported_values.keys())
            )
            raise ValueError(message)

    if isinstance(item, (float, int)):
        return item

    return default


def string_to_bool(string):
    return string.lower() in ['yes', 'true', 't', '1']


def get_key_by_value(container, value):
    for k, v in container.items():
        if v == value:
            return k

    return None


def format_key(key):
    return '--{}'.format(key)


def to_lower_register(str_list):
    return list(map(lambda item: item.lower() if item else None, str_list))
