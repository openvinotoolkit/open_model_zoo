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
import errno
import itertools
import json
import os
import pathlib
import numpy as np

from .representation import BaseRepresentation


def concat_lists(*lists):
    return list(itertools.chain(*lists))


def check_exists(entry: str):
    path = pathlib.Path(entry).absolute()
    if not path.exists():
        raise FileNotFoundError('{}: {}'.format(os.strerror(errno.ENOENT), path))
    return path


def read_annotation(annotation_file: str):
    annotation = pathlib.Path(annotation_file).absolute()
    if not annotation.is_file():
        raise FileNotFoundError('{}: {}'.format(os.strerror(errno.ENOENT), annotation))

    result = []
    with annotation.open('rb') as file:
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
    if isinstance(obj, type):
        cls = obj
    else:
        cls = obj.__class__
    base = cls.__bases__[0] if base is None else base
    obj_attr = getattr(cls, attribute_name, None)
    base_attr = getattr(base, attribute_name, None)
    return obj_attr is not None and obj_attr != base_attr


def enum_values(enum):
    return [member.value for member in enum]


def get_size_from_config(config):
    if 'size' in config:
        return config['size'], config['size']
    if contains_all(config, ('dst_width', 'dst_height')):
        return config['dst_height'], config['dst_width']
    raise ValueError('Either size or dst_width and dst_height required')


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
    result_values = []
    result_names = []
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
