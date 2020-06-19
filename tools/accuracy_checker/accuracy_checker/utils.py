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

import csv
import errno
import itertools
import json
import os
import pickle
from enum import Enum

from pathlib import Path
from typing import Union
from warnings import warn
from collections.abc import MutableSet

import numpy as np
import yaml

try:
    import lxml.etree as et
except ImportError:
    import xml.etree.cElementTree as et

try:
    from shapely.geometry.polygon import Polygon
except ImportError:
    Polygon = None

try:
    from yamlloader.ordereddict import Loader as orddict_loader
except ImportError:
    orddict_loader = None


def concat_lists(*lists):
    return list(itertools.chain(*lists))


def get_path(entry: Union[str, Path], is_directory=False, check_exists=True, file_or_directory=False):
    try:
        path = Path(entry)
    except TypeError:
        raise TypeError('"{}" is expected to be a path-like'.format(entry))

    if not check_exists:
        return path

    # pathlib.Path.exists throws an exception in case of broken symlink
    if not os.path.exists(str(path)):
        raise FileNotFoundError('{}: {}'.format(os.strerror(errno.ENOENT), path))

    if not file_or_directory:
        if is_directory and not path.is_dir():
            raise NotADirectoryError('{}: {}'.format(os.strerror(errno.ENOTDIR), path))

        # if it exists it is either file (or valid symlink to file) or directory (or valid symlink to directory)
        if not is_directory and not path.is_file():
            raise IsADirectoryError('{}: {}'.format(os.strerror(errno.EISDIR), path))

    return path


def contains_all(container, *args):
    sequence = set(container)

    for arg in args:
        if len(sequence.intersection(arg)) != len(arg):
            return False

    return True


def contains_any(container, *args):
    sequence = set(container)

    for arg in args:
        if sequence.intersection(arg):
            return True

    return False


def string_to_tuple(string, casting_type=float):
    processed = string.replace(' ', '')
    processed = processed.replace('(', '')
    processed = processed.replace(')', '')
    processed = processed.split(',')

    return tuple([casting_type(entry) for entry in processed]) if not casting_type is None else tuple(processed)


def string_to_list(string):
    processed = string.replace(' ', '')
    processed = processed.replace('[', '')
    processed = processed.replace(']', '')
    processed = processed.split(',')

    return list(entry for entry in processed)


class JSONDecoderWithAutoConversion(json.JSONDecoder):
    """
    Custom json decoder to convert all strings into numbers (int, float) during reading json file.
    """

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
    return {key: value for key, value in dict_.items() if key in key_subset}


def zipped_transform(fn, *iterables, inplace=False):
    result = (iterables if inplace else tuple([] for _ in range(len(iterables))))
    updater = (list.__setitem__ if inplace else lambda container, _, entry: container.append(entry))

    for idx, values in enumerate(zip(*iterables)):
        iter_res = fn(*values)
        if not iter_res:
            continue

        for dst, res in zip(result, iter_res):
            updater(dst, idx, res)

    return result


def overrides(obj, attribute_name, base=None):
    cls = obj if isinstance(obj, type) else obj.__class__

    base = base or cls.__bases__[0]
    obj_attr = getattr(cls, attribute_name, None)
    base_attr = getattr(base, attribute_name, None)

    return obj_attr and obj_attr != base_attr


def enum_values(enum):
    return [member.value for member in enum]


def get_size_from_config(config, allow_none=False):
    if contains_all(config, ('size', 'dst_width', 'dst_height')):
        warn('All parameters: size, dst_width, dst_height are provided. Size will be used. '
             'You should specify only size or pair values des_width, dst_height in config.')
    if 'size' in config:
        return config['size'], config['size']
    if contains_all(config, ('dst_width', 'dst_height')):
        return config['dst_height'], config['dst_width']
    if not allow_none:
        raise ValueError('Either size or dst_width and dst_height required')

    return None, None


def get_size_3d_from_config(config, allow_none=False):
    if contains_all(config, ('size', 'dst_width', 'dst_height', 'dst_volume')):
        warn('All parameters: size, dst_width, dst_height, dst_volume are provided. Size will be used. '
             'You should specify only size or three values des_width, dst_height, dst_volume in config.')
    if 'size' in config:
        return config['size'], config['size'], config['size']
    if contains_all(config, ('dst_width', 'dst_height', 'dst_volume')):
        return config['dst_height'], config['dst_width'], config['dst_volume']
    if not allow_none:
        raise ValueError('Either size or dst_width and dst_height required')

    return config.get('dst_height'), config.get('dst_width'), config.get('dst_volume')


def parse_inputs(inputs_entry):
    inputs = []
    for inp in inputs_entry:
        value = inp.get('value')
        shape = inp.get('shape')
        new_input = {'name': inp['name']}
        if value is not None:
            new_input['value'] = np.array(value) if isinstance(value, list) else value
        if shape is not None:
            new_input['shape'] = shape

        inputs.append(new_input)
    return inputs


def in_interval(value, interval):
    minimum = interval[0]
    maximum = interval[1] if len(interval) >= 2 else None

    if not maximum:
        return minimum <= value

    return minimum <= value < maximum


def is_config_input(input_name, config_inputs):
    for config_input in config_inputs:
        if config_input['name'] == input_name:
            return True
    return False


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
    for representation_type in representation_types:
        if type(representation).__name__ == representation_type.__name__:
            return True
    return False


def is_single_metric_source(source):
    if not source:
        return False

    return np.size(source.split(',')) == 1


def read_txt(file: Union[str, Path], sep='\n', **kwargs):
    def is_empty(string):
        return not string or string.isspace()

    with get_path(file).open(**kwargs) as content:
        content = content.read().split(sep)
        content = list(filter(lambda string: not is_empty(string), content))

        return list(map(str.strip, content))


def read_xml(file: Union[str, Path], *args, **kwargs):
    return et.parse(str(get_path(file)), *args, **kwargs).getroot()


def read_json(file: Union[str, Path], *args, **kwargs):
    with get_path(file).open() as content:
        return json.load(content, *args, **kwargs)


def read_pickle(file: Union[str, Path], *args, **kwargs):
    with get_path(file).open('rb') as content:
        return pickle.load(content, *args, **kwargs)


def read_yaml(file: Union[str, Path], *args, ordered=True, **kwargs):
    with get_path(file).open() as content:
        loader = orddict_loader or yaml.SafeLoader if ordered else yaml.SafeLoader
        if not orddict_loader and ordered:
            warn('yamlloader is not installed. YAML files order is not preserved. it can be sufficient for some cases')
        return yaml.load(content, *args, Loader=loader, **kwargs)


def read_csv(file: Union[str, Path], *args, **kwargs):
    with get_path(file).open() as content:
        return list(csv.DictReader(content, *args, **kwargs))


def extract_image_representations(image_representations):
    images = [rep.data for rep in image_representations]
    meta = [rep.metadata for rep in image_representations]

    return images, meta


def convert_bboxes_xywh_to_x1y1x2y2(x_coord, y_coord, width, height):
    return x_coord, y_coord, x_coord + width, y_coord + height


def get_or_parse_value(item, supported_values=None, default=None, casting_type=float):
    if isinstance(item, str):
        item = item.lower()
        if supported_values and item in supported_values:
            return supported_values[item]

        try:
            return string_to_tuple(item, casting_type=casting_type)
        except ValueError:
            message = 'Invalid value "{}", expected {}list of values'.format(
                item,
                'one of precomputed: ({}) or '.format(', '.join(supported_values.keys())) if supported_values else ''
            )
            raise ValueError(message)

    if isinstance(item, (float, int)):
        return (casting_type(item), )

    if isinstance(item, list):
        return item

    return default


def cast_to_bool(entry):
    if isinstance(entry, str):
        return entry.lower() in ['yes', 'true', 't', '1']
    return bool(entry)


def get_key_by_value(container, target):
    for key, value in container.items():
        if value == target:
            return key

    return None


def format_key(key):
    return '--{}'.format(key)


def to_lower_register(str_list):
    return list(map(lambda item: item.lower() if item else None, str_list))


def polygon_from_points(points):
    if Polygon is None:
        raise ValueError('shapely is not installed, please install it')
    return Polygon(points)


def remove_difficult(difficult, indexes):
    new_difficult = []
    decrementor = 0
    id_difficult = 0
    id_removed = 0
    while id_difficult < len(difficult) and id_removed < len(indexes):
        if difficult[id_difficult] < indexes[id_removed]:
            new_difficult.append(difficult[id_difficult] - decrementor)
            id_difficult += 1
        else:
            decrementor += 1
            id_removed += 1

    return new_difficult


def convert_to_range(entry):
    entry_range = entry
    if isinstance(entry, str):
        entry_range = string_to_tuple(entry_range)
    elif not isinstance(entry_range, tuple) and not isinstance(entry_range, list):
        entry_range = [entry_range]

    return entry_range


def add_input_shape_to_meta(meta, shape):
    meta['input_shape'] = shape
    return meta


def set_image_metadata(annotation, images):
    image_sizes = []
    data = images.data
    if not isinstance(data, list):
        data = [data]
    for image in data:
        data_shape = image.shape if not np.isscalar(image) else 1
        image_sizes.append(data_shape)
    annotation.set_image_size(image_sizes)

    return annotation, images


def get_indexs(container, element):
    return [index for index, container_element in enumerate(container) if container_element == element]


def find_nearest(array, value, mode=None):
    if not array:
        return -1
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if mode == 'less':
        return idx - 1 if array[idx] > value else idx
    if mode == 'more':
        return idx + 1 if array[idx] < value else idx
    return idx


class OrderedSet(MutableSet):
    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]
        self.map = {}
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:
            key, prev_value, next_value = self.map.pop(key)
            prev_value[2] = next_value
            next_value[1] = prev_value

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '{}()'.format(self.__class__.__name__,)
        return '{}({})'.format(self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)


def get_parameter_value_from_config(config, parameters, key):
    if key not in parameters.keys():
        return None
    field = parameters[key]
    value = config.get(key, field.default)
    field.validate(value)
    return value


def check_file_existence(file):
    try:
        get_path(file)
        return True
    except (FileNotFoundError, IsADirectoryError):
        return False


class Color(Enum):
    PASSED = 0
    FAILED = 1


def color_format(s, color=Color.PASSED):
    if color == Color.PASSED:
        return "\x1b[0;32m{}\x1b[0m".format(s)
    return "\x1b[0;31m{}\x1b[0m".format(s)


def softmax(x):
    return np.exp(x) / sum(np.exp(x))
