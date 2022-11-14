"""
Copyright (c) 2018-2022 Intel Corporation

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
import pickle # nosec - disable B403:import-pickle check
import struct
import sys
import zlib
import re
from enum import Enum

from pathlib import Path
from typing import Union
from warnings import warn
from collections.abc import MutableSet, Sequence
from io import BytesIO

import defusedxml.ElementTree as et
import numpy as np
import yaml

from . import __version__

try:
    from shapely.geometry.polygon import Polygon
except ImportError:
    Polygon = None


def concat_lists(*lists):
    return list(itertools.chain(*lists))


def get_path(entry: Union[str, Path], is_directory=False, check_exists=True, file_or_directory=False):
    try:
        path = Path(entry)
    except TypeError as type_err:
        raise TypeError('"{}" is expected to be a path-like'.format(entry)) from type_err
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
        if sequence.intersection(arg) != set(arg):
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
    processed = filter(lambda x: x, processed)
    return tuple(map(casting_type, processed)) if casting_type else tuple(processed)


def string_to_list(string):
    processed = string.replace(' ', '')
    processed = processed.replace('[', '')
    processed = processed.replace(']', '')
    processed = processed.split(',')
    return processed


def validate_print_interval(value, min_value=0, max_value=None):
    if value <= min_value:
        raise ValueError('{} less than minimum required {}'.format(value, min_value))
    if max_value and value >= max_value:
        raise ValueError('{} greater than maximum required {}'.format(value, max_value))

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


def is_path(data):
    return isinstance(data, (Path, str))


def read_txt(file: Union[str, Path], sep='\n', ignore_space=False, **kwargs):
    def is_empty(string):
        emptyness = not string
        if not ignore_space:
            emptyness = emptyness or string.isspace()
        return emptyness

    with get_path(file).open(**kwargs) as content:
        content = content.read().split(sep)
        content = list(filter(lambda string: not is_empty(string), content))
        if not ignore_space:
            content = list(map(str.strip, content))
        return content


def read_xml(file: Union[str, Path], *args, **kwargs):
    return et.parse(str(get_path(file)), *args, **kwargs).getroot()


def read_json(file: Union[str, Path], *args, **kwargs):
    with get_path(file).open() as content:
        return json.load(content, *args, **kwargs)


def read_pickle(file: Union[str, Path], *args, **kwargs):
    with get_path(file).open('rb') as content:
        return pickle.load(content, *args, **kwargs) # nosec - disable B301:pickle check


class RenameUnpickler(pickle.Unpickler): # nosec - disable B301:pickle check
    def __init__(self, file, renaming_mapping, *args, **kwargs):
        self.renaming_mapping = renaming_mapping
        super().__init__(file, *args, **kwargs)

    def find_class(self, module, name):
        renamed_module = module
        for old_module, new_module in self.renaming_mapping.items():
            if module.startswith(old_module):
                if isinstance(new_module, list):
                    for nm in new_module:
                        try:
                            renamed_module = module.replace(old_module, nm, 1)
                            res = super().find_class(renamed_module, name)
                            return res
                        except ModuleNotFoundError:
                            continue
                else:
                    renamed_module = module.replace(old_module, new_module, 1)
        return super().find_class(renamed_module, name)


def read_pickle_with_renaming(file: Union[str, Path], renaming_mapping, *args, **kwargs):
    with get_path(file).open('rb') as content:
        return RenameUnpickler(content, renaming_mapping, *args, **kwargs).load()


def read_yaml(file: Union[str, Path], *args, **kwargs):
    with get_path(file).open() as content:
        return yaml.safe_load(content, *args, **kwargs)


def read_csv(file: Union[str, Path], *args, is_dict=True, **kwargs):
    with get_path(file).open(encoding='utf-8') as content:
        if is_dict:
            return list(csv.DictReader(content, *args, **kwargs))
        return list(csv.reader(content, *args, **kwargs))


def extract_image_representations(image_representations, meta_only=False):
    meta = [rep.metadata for rep in image_representations]
    if meta_only:
        return meta
    images = [rep.data for rep in image_representations]
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
        except ValueError as value_err:
            message = 'Invalid value "{}", expected {}list of values'.format(
                item,
                'one of precomputed: ({}) or '.format(', '.join(supported_values.keys())) if supported_values else ''
            )
            raise ValueError(message) from value_err
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
    image_sizes = get_data_shapes(images)
    annotation.set_image_size(image_sizes)
    return annotation, images


def get_data_shapes(images):
    image_sizes = []
    data = images.data
    if not isinstance(data, list):
        data = [data]
    for image in data:
        data_shape = np.shape(image) if not np.isscalar(image) else 1
        image_sizes.append(data_shape)
    return image_sizes


def is_image(data_shape):
    if len(data_shape) not in [2, 3]:
        return False
    if len(data_shape) == 3:
        if data_shape[-1] not in [1, 3, 4]:
            return False
    return True


def finalize_image_shape(dst_h, dst_w, initial_shape):
    if len(initial_shape) == 2:
        return (dst_h, dst_w)
    return tuple([dst_h, dst_w] + list(initial_shape[2:]))


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

    def add(self, value):
        if value not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[value] = [value, curr, end]

    def discard(self, value):
        if value in self.map:
            value, prev_value, next_value = self.map.pop(value)
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


def is_relative_to(path, *other):
    try:
        Path(path).relative_to(*other)
        return True
    except ValueError:
        return False


def get_parameter_value_from_config(config, parameters, key):
    if key not in parameters.keys():
        return None
    field = parameters[key]
    value = config.get(key, field.default)
    field.validate(value)
    data_type = field.type
    if value is not None and data_type is not None:
        value = data_type(value)
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


def softmax(x, axis=0):
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)


def is_iterable(maybe_iterable):
    try:
        iter(maybe_iterable)
        return True
    except TypeError:
        return False


class ParseError(Exception):
    pass


class MatlabDataReader():
    def __init__(self):
        self.asstr = lambda b: b.decode('latin1')
        self.etypes = {
            'miINT8': {'n': 1, 'fmt': 'b'},
            'miUINT8': {'n': 2, 'fmt': 'B'},
            'miINT16': {'n': 3, 'fmt': 'h'},
            'miUINT16': {'n': 4, 'fmt': 'H'},
            'miINT32': {'n': 5, 'fmt': 'i'},
            'miUINT32': {'n': 6, 'fmt': 'I'},
            'miSINGLE': {'n': 7, 'fmt': 'f'},
            'miDOUBLE': {'n': 9, 'fmt': 'd'},
            'miINT64': {'n': 12, 'fmt': 'q'},
            'miUINT64': {'n': 13, 'fmt': 'Q'},
            'miMATRIX': {'n': 14},
            'miCOMPRESSED': {'n': 15},
            'miUTF8': {'n': 16, 'fmt': 's'},
            'miUTF16': {'n': 17, 'fmt': 's'},
            'miUTF32': {'n': 18, 'fmt': 's'}
        }
        self.inv_etypes = {v['n']: k for k, v in self.etypes.items()}
        self.mclasses = {
            'mxCELL_CLASS': 1,
            'mxSTRUCT_CLASS': 2,
            'mxOBJECT_CLASS': 3,
            'mxCHAR_CLASS': 4,
            'mxSPARSE_CLASS': 5,
            'mxDOUBLE_CLASS': 6,
            'mxSINGLE_CLASS': 7,
            'mxINT8_CLASS': 8,
            'mxUINT8_CLASS': 9,
            'mxINT16_CLASS': 10,
            'mxUINT16_CLASS': 11,
            'mxINT32_CLASS': 12,
            'mxUINT32_CLASS': 13,
            'mxINT64_CLASS': 14,
            'mxUINT64_CLASS': 15,
            'mxFUNCTION_CLASS': 16,
            'mxOPAQUE_CLASS': 17,
            'mxOBJECT_CLASS_FROM_MATRIX_H': 18
        }
        self.numeric_class_etypes = {
            'mxDOUBLE_CLASS': 'miDOUBLE',
            'mxSINGLE_CLASS': 'miSINGLE',
            'mxINT8_CLASS': 'miINT8',
            'mxUINT8_CLASS': 'miUINT8',
            'mxINT16_CLASS': 'miINT16',
            'mxUINT16_CLASS': 'miUINT16',
            'mxINT32_CLASS': 'miINT32',
            'mxUINT32_CLASS': 'miUINT32',
            'mxINT64_CLASS': 'miINT64',
            'mxUINT64_CLASS': 'miUINT64'
        }
        self.inv_mclasses = {v: k for k, v in self.mclasses.items()}
        self.compressed_numeric = ['miINT32', 'miUINT16', 'miINT16', 'miUINT8']

    def read_var_header(self, fd, endian):
        mtpn, num_bytes = self._unpack(endian, 'II', fd.read(8))
        next_pos = fd.tell() + num_bytes

        if mtpn == self.etypes['miCOMPRESSED']['n']:
            data = fd.read(num_bytes)
            dcor = zlib.decompressobj()
            fd_var = BytesIO(dcor.decompress(data))
            del data
            fd = fd_var
            if dcor.flush() != b'':
                raise ParseError('Error in compressed data.')
            mtpn, num_bytes = self._unpack(endian, 'II', fd.read(8))
        if mtpn != self.etypes['miMATRIX']['n']:
            raise ParseError('Expecting miMATRIX type number {}, '
                             'got {}'.format(self.etypes['miMATRIX']['n'], mtpn))
        header = self._read_header(fd, endian)
        return header, next_pos, fd

    def read_var_array(self, fd, endian, header):
        mc = self.inv_mclasses[header['mclass']]
        if mc == 'mxSPARSE_CLASS':
            raise ParseError('Sparse matrices not supported')
        if mc == 'mxOBJECT_CLASS':
            raise ParseError('Object classes not supported')
        if mc == 'mxFUNCTION_CLASS':
            raise ParseError('Function classes not supported')
        if mc == 'mxOPAQUE_CLASS':
            raise ParseError('Anonymous function classes not supported')
        if mc in self.numeric_class_etypes:
            return self._read_numeric_array(
                fd, endian, header,
                set(self.compressed_numeric).union([self.numeric_class_etypes[mc]])
            )
        if mc == 'mxCHAR_CLASS':
            return self._read_char_array(fd, endian, header)
        if mc == 'mxCELL_CLASS':
            return self._read_cell_array(fd, endian, header)
        if mc == 'mxSTRUCT_CLASS':
            return self._read_struct_array(fd, endian, header)
        return None

    def _read_element_tag(self, fd, endian):
        data = fd.read(8)
        mtpn = self._unpack(endian, 'I', data[:4])
        num_bytes = mtpn >> 16
        if num_bytes > 0:
            mtpn = mtpn & 0xFFFF
            if num_bytes > 4:
                raise ParseError('Error parsing Small Data Element (SDE) '
                                 'formatted data')
            data = data[4:4 + num_bytes]
        else:
            num_bytes = self._unpack(endian, 'I', data[4:])
            data = None
        return (mtpn, num_bytes, data)

    def _read_elements(self, fd, endian, mtps, is_name=False):
        mtpn, num_bytes, data = self._read_element_tag(fd, endian)
        if mtps and mtpn not in [self.etypes[mtp]['n'] for mtp in mtps]:
            raise ParseError('Got type {}, expected {}'.format(
                mtpn, ' / '.join('{} ({})'.format(
                    self.etypes[mtp]['n'], mtp) for mtp in mtps)))
        if not data:
            data = fd.read(num_bytes)
            mod8 = num_bytes % 8
            if mod8:
                fd.seek(8 - mod8, 1)
        if is_name:
            fmt = 's'
            val = [self._unpack(endian, fmt, s)
                   for s in data.split(b'\0') if s]
            if len(val) == 0:
                val = ''
            elif len(val) == 1:
                val = self.asstr(val[0])
            else:
                val = [self.asstr(s) for s in val]
        else:
            fmt = self.etypes[self.inv_etypes[mtpn]]['fmt']
            val = self._unpack(endian, fmt, data)
        return val

    def _read_header(self, fd, endian):
        flag_class, nzmax = self._read_elements(fd, endian, ['miUINT32'])
        header = {
            'mclass': flag_class & 0x0FF,
            'is_logical': (flag_class >> 9 & 1) == 1,
            'is_global': (flag_class >> 10 & 1) == 1,
            'is_complex': (flag_class >> 11 & 1) == 1,
            'nzmax': nzmax
        }
        header['dims'] = self._read_elements(fd, endian, ['miINT32'])
        header['n_dims'] = len(header['dims'])
        if header['n_dims'] != 2:
            raise ParseError('Only matrices with dimension 2 are supported.')
        header['name'] = self._read_elements(fd, endian, ['miINT8'], is_name=True)
        return header

    def _read_numeric_array(self, fd, endian, header, data_etypes):
        if header['is_complex']:
            raise ParseError('Complex arrays are not supported')
        data = self._read_elements(fd, endian, data_etypes)
        if not isinstance(data, Sequence):
            return data
        rowcount = header['dims'][0]
        colcount = header['dims'][1]
        array = [[data[c * rowcount + r] for c in range(colcount)]
                 for r in range(rowcount)]
        return self._squeeze(array)

    def _read_cell_array(self, fd, endian, header):
        array = [[] for i in range(header['dims'][0])]
        for row in range(header['dims'][0]):
            for _col in range(header['dims'][1]):
                vheader, next_pos, fd_var = self.read_var_header(fd, endian)
                varray = self.read_var_array(fd_var, endian, vheader)
                array[row].append(varray)
                fd.seek(next_pos)
        if header['dims'][0] == 1:
            return self._squeeze(array[0])
        return self._squeeze(array)

    def _read_struct_array(self, fd, endian, header):
        field_name_length = self._read_elements(fd, endian, ['miINT32'])
        if field_name_length > 32:
            raise ParseError('Unexpected field name length: {}'.format(
                field_name_length))

        fields = self._read_elements(fd, endian, ['miINT8'], is_name=True)
        if isinstance(fields, str):
            fields = [fields]

        array = {}
        for row in range(header['dims'][0]):
            for _col in range(header['dims'][1]):
                for field in fields:
                    vheader, next_pos, fd_var = self.read_var_header(fd, endian)
                    data = self.read_var_array(fd_var, endian, vheader)
                    if field not in array:
                        array[field] = [[] for _ in range(header['dims'][0])]
                    array[field][row].append(data)
                    fd.seek(next_pos)
        for field in fields:
            rows = array[field]
            for i in range(header['dims'][0]):
                rows[i] = self._squeeze(rows[i])
            array[field] = self._squeeze(array[field])
        return array

    def _read_char_array(self, fd, endian, header):
        array = self._read_numeric_array(fd, endian, header, ['miUTF8', 'miUTF16'])
        if header['dims'][0] > 1:
            array = [self.asstr(bytearray(i)) for i in array]
        else:
            array = self.asstr(bytearray(array))
        return array

    @staticmethod
    def _squeeze(array):
        if len(array) == 1:
            array = array[0]
        return array

    @staticmethod
    def _unpack(endian, fmt, data):
        if fmt == 's':
            val = struct.unpack(''.join([endian, str(len(data)), 's']),
                                data)[0]
        else:
            num = len(data) // struct.calcsize(fmt)
            val = struct.unpack(''.join([endian, str(num), fmt]), data)
            if len(val) == 1:
                val = val[0]
        return val


def loadmat(filename):
    def eof(fd):
        b = fd.read(1)
        end = len(b) == 0
        if not end:
            curpos = fd.tell()
            fd.seek(curpos - 1)
        return end

    fd = open(filename, 'rb') # pylint: disable=R1732
    fd.seek(124)
    tst_str = fd.read(4)
    little_endian = (tst_str[2:4] == b'IM')
    endian = ''
    if sys.byteorder == 'little' and not little_endian:
        endian = '>'
    if sys.byteorder == 'big' and little_endian:
        endian = '<'
    maj_ind = int(little_endian)
    maj_val = tst_str[maj_ind]
    if maj_val != 1:
        raise ParseError('Can only read from Matlab level 5 MAT-files')
    mdict = {}
    reader = MatlabDataReader()

    while not eof(fd):
        hdr, next_position, fd_var = reader.read_var_header(fd, endian)
        name = hdr['name']
        if name in mdict:
            raise ParseError('Duplicate variable name "{}" in mat file.'
                             .format(name))
        mdict[name] = reader.read_var_array(fd_var, endian, hdr)
        fd.seek(next_position)
    fd.close()
    return mdict


class UnsupportedPackage:
    def __init__(self, package, message):
        self.package = package
        self.msg = message

    def raise_error(self, provider):
        msg = "{package} is not installed. Please install it before using {provider}.\n{message}".format(
            provider=provider, package=self.package, message=self.msg)
        raise ImportError(msg)

    def __call__(self, *args, **kwargs):
        self.raise_error('')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_layer_name(layer_name, prefix, with_prefix):
    return prefix + layer_name if with_prefix else layer_name.split(prefix, 1)[-1]


def convert_xctr_yctr_w_h_to_x1y1x2y2(x, y, width, height):
    x1, y1 = (x - width / 2), (y - height / 2)
    x2, y2 = (x + width / 2), (y + height / 2)
    return x1, y1, x2, y2


def init_telemetry():
    try:
        import openvino_telemetry as tm # pylint:disable=C0415
    except ImportError:
        return None
    try:
        telemetry = tm.Telemetry(tid='UA-17808594-29', app_name='Accuracy Checker', app_version=__version__)
        return telemetry
    except Exception: # pylint:disable=W0703
        return None


def send_telemetry_event(tm, *args, **kwargs):
    if tm is None:
        return
    try:
        tm.send_event('ac', *args, **kwargs)
    except Exception: # pylint:disable=W0703
        pass
    return


def start_telemetry():
    tm = init_telemetry()
    if tm:
        try:
            tm.start_session('ac')
        except Exception:  # pylint:disable=W0703
            pass
    return tm


def end_telemetry(tm):
    if tm:
        try:
            tm.end_session('ac')
            tm.force_shutdown(1.0)
        except Exception: # pylint:disable=W0703
            pass


def parse_partial_shape(partial_shape):
    ps = str(partial_shape)
    if ps[0] == '[' and ps[-1] == ']':
        ps = ps[1:-1]
    preprocessed = ps.replace('{', '(').replace('}', ')').replace('?', '-1')
    if '[' not in preprocessed:
        preprocessed = preprocessed.replace('(', '').replace(')', '')
        if '..' in preprocessed:
            shape_list = []
            for dim in preprocessed.split(','):
                if '..' in dim:
                    shape_list.append(string_to_tuple(dim.replace('..', ','), casting_type=int))
                else:
                    shape_list.append(int(dim))
            return shape_list
        return string_to_tuple(preprocessed, casting_type=int)
    shape_list = []
    s_pos = 0
    e_pos = len(preprocessed)
    while s_pos <= e_pos:
        open_brace = preprocessed.find('[', s_pos, e_pos)
        if open_brace == -1:
            shape_list.extend(string_to_tuple(preprocessed[s_pos:], casting_type=int))
            break
        if open_brace != s_pos:
            shape_list.extend(string_to_tuple(preprocessed[:open_brace], casting_type=int))
        close_brace = preprocessed.find(']', open_brace, e_pos)
        shape_range = preprocessed[open_brace + 1:close_brace]
        shape_list.append(string_to_tuple(shape_range, casting_type=int))
        s_pos = min(close_brace + 2, e_pos)
    return shape_list


def postprocess_output_name(
    output_name, outputs, suffix=('/sink_port_', ':'), additional_mapping=None, raise_error=True
):
    suffixes = [suffix] if isinstance(suffix, str) else suffix
    outputs = outputs[0] if isinstance(outputs, list) else outputs
    if output_name in outputs:
        return output_name
    if additional_mapping and output_name in additional_mapping:
        return additional_mapping[output_name]
    for suffix_ in suffixes:
        matches = re.findall(r'{}\d+'.format(suffix_), output_name)
        if matches:
            preprocessed_output_name = output_name.replace(matches[0], '')
        else:
            suffix_regex = re.compile(output_name + suffix_ + r'\d+')
            preprocessed_output_name = [layer_name for layer_name in outputs if suffix_regex.match(layer_name)]
            preprocessed_output_name = preprocessed_output_name[0] if preprocessed_output_name else ''
        if preprocessed_output_name in outputs:
            return preprocessed_output_name
    if raise_error:
        raise ValueError('Output name: {} not found'.format(output_name))
    return output_name

# pylint:disable=C0415,W0611
def ov_new_api_available():
    try:
        import openvino # noqa: F401
    except ImportError:
        return None
    try:
        from openvino.runtime import Core # noqa: F401
        return True
    except ImportError:
        return False
