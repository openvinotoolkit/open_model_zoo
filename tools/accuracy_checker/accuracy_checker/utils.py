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

import csv
import errno
import itertools
import json
import os
import pickle
import struct
import sys
import zlib
from enum import Enum

from pathlib import Path
from typing import Union
from warnings import warn
from collections.abc import MutableSet, Sequence
from io import BytesIO

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


def read_yaml(file: Union[str, Path], *args, **kwargs):
    with get_path(file).open() as content:
        return yaml.safe_load(content, *args, **kwargs)


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


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


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
        self.inv_etypes = dict((v['n'], k) for k, v in self.etypes.items())
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
        self.inv_mclasses = dict((v, k) for k, v in self.mclasses.items())
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
        array = [list(data[c * rowcount + r] for c in range(colcount))
                 for r in range(rowcount)]
        return self._squeeze(array)

    def _read_cell_array(self, fd, endian, header):
        array = [list() for i in range(header['dims'][0])]
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

        empty = lambda: [list() for i in range(header['dims'][0])]
        array = {}
        for row in range(header['dims'][0]):
            for _col in range(header['dims'][1]):
                for field in fields:
                    vheader, next_pos, fd_var = self.read_var_header(fd, endian)
                    data = self.read_var_array(fd_var, endian, vheader)
                    if field not in array:
                        array[field] = empty()
                    array[field][row].append(data)
                    fd.seek(next_pos)
        for field in fields:
            rows = array[field]
            for i in range(header['dims'][0]):
                rows[i] = self._squeeze(rows[i])
            array[field] = self._squeeze(array[field])
        return array

    def _read_char_array(self, fd, endian, header):
        array = self._read_numeric_array(fd, endian, header, ['miUTF8'])
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

    fd = open(filename, 'rb')

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
