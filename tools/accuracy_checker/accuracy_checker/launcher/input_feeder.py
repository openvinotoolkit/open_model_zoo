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

import re
import numpy as np

from ..config import ConfigError
from ..utils import extract_image_representations
from ..data_readers import MultiFramesInputIdentifier

LAYER_LAYOUT_TO_IMAGE_LAYOUT = {
    'NCHW': [0, 3, 1, 2],
    'NHWC': [0, 1, 2, 3],
    'NCWH': [0, 3, 2, 1],
    'NWHC': [0, 2, 1, 3],
    'NCDHW': [0, 4, 1, 2, 3],
    'NDCHW': [0, 1, 4, 2, 3],
    'NDHWC': [0, 1, 2, 3, 4],
    'NDCWH': [0, 1, 4, 3, 2],
    'NCHWD': [0, 2, 3, 4, 1],
    'NC': [0, 1],
    'CN': [1, 0]
}

DIM_IDS_TO_LAYOUT = {
    (0, 3, 1, 2): 'NCHW',
    (0, 1, 2, 3): 'NHWC',
    (0, 4, 1, 2, 3): 'NCDHW',
    (0, 1, 4, 2, 3): 'NDHWC',
    (0, 1, 4, 3, 2): 'NDCWH',
    (0, 1): 'NC',
    (1, 0): 'CN'
}

PRECISION_TO_DTYPE = {
    'FP32': np.float32,  # float
    'FP16': np.float16,  # signed short
    'U8': np.uint8,  # unsigned char
    'U16': np.uint16,  # unsigned short
    'I8': np.int8,  # signed char
    'I16': np.int16,  # signed short
    'I32': np.int32,  # signed int
    'I64': np.int64,  # signed long int
}


class InputFeeder:
    def __init__(self, inputs_config, network_inputs, prepare_input_data=None, default_layout='NCHW', dummy=False):
        def fit_to_input(data, input_layer_name, layout, precision):
            if len(np.shape(data)) == 4:
                data = np.transpose(data, layout)
            else:
                data = np.array(data)
            return data.astype(precision) if precision else data

        self.input_transform_func = prepare_input_data or fit_to_input
        self.network_inputs = network_inputs or []
        self.default_layout = default_layout
        self.dummy = dummy
        self.configure(inputs_config)

    def configure(self, inputs_config):
        if not self.dummy:
            parsing_results = self._parse_inputs_config(inputs_config, self.default_layout)
            self.const_inputs, self.non_constant_inputs, self.inputs_mapping = parsing_results[:3]
            self.image_info_inputs, self.lstm_inputs, self.layouts_mapping, self.precision_mapping = parsing_results[3:]
            if not self.non_constant_inputs:
                raise ConfigError('Network should contain at least one layer for setting variable data.')

    def _fill_image_info_inputs(self, data_representation_batch):
        def prepare_image_info(image_sizes_batch):
            image_info = []
            for image_size in image_sizes_batch:
                if np.isscalar(image_size) or isinstance(image_size, list):
                    image_info.append(image_size)
                    continue

                height, width = image_size[:2]
                image_info.append([height, width, 1])

            return image_info
        _, meta_batch = extract_image_representations(data_representation_batch)
        if 'image_info' in meta_batch[0]:
            image_info_data = [meta['image_info'] for meta in meta_batch]
            return {image_info_input: image_info_data for image_info_input in self.image_info_inputs}
        image_sizes = [meta['image_size'] for meta in meta_batch]
        image_info_data = prepare_image_info(image_sizes)
        image_infos = {image_info_input: image_info_data for image_info_input in self.image_info_inputs}

        return image_infos

    def fill_non_constant_inputs(self, data_representation_batch):
        filled_inputs = {}
        if self.image_info_inputs:
            image_info_inputs = self._fill_image_info_inputs(data_representation_batch)
            filled_inputs = {**image_info_inputs}
        for input_layer in self.non_constant_inputs:
            input_regex = None
            input_batch = []
            if self.inputs_mapping:
                input_regex = self.inputs_mapping[input_layer]
            for data_representation in data_representation_batch:
                input_data = None
                identifiers = data_representation.identifier
                data = data_representation.data
                if not isinstance(identifiers, list) and input_regex is None:
                    input_data = data
                    input_batch.append(input_data)
                    continue

                if input_regex is None:
                    raise ConfigError('Impossible to choose correct data for layer {}.'
                                      'Please provide regular expression for matching in config.'.format(input_layer))
                if isinstance(identifiers, MultiFramesInputIdentifier):
                    input_id_order = {
                        input_index: frame_id for frame_id, input_index in enumerate(identifiers.input_id)
                    }
                    input_data = data[input_id_order[input_regex]]
                else:
                    data = [data] if np.isscalar(identifiers) else data
                    identifiers = [identifiers] if np.isscalar(identifiers) else identifiers
                    for identifier, data_value in zip(identifiers, data):
                        if input_regex.match(identifier):
                            input_data = data_value
                            break
                if input_data is None:
                    raise ConfigError('Suitable data for filling layer {} not found'.format(input_layer))
                input_batch.append(input_data)

            filled_inputs[input_layer] = input_batch

        return self._transform_batch(filled_inputs, extract_image_representations(data_representation_batch)[1])

    def fill_inputs(self, data_representation_batch):
        if self.dummy:
            return []
        inputs = self.fill_non_constant_inputs(data_representation_batch)
        for infer_inputs in inputs:
            infer_inputs.update(self.const_inputs)
        return inputs

    def _parse_inputs_config(self, inputs_entry, default_layout='NCHW'):
        def get_layer_precision(input_config, input_name):
            if 'precision' not in input_config:
                return None
            input_precision = PRECISION_TO_DTYPE.get(input_config['precision'])
            if input_precision is None:
                raise ConfigError("unsupported precision {} for layer {}".format(input_config['precision'], input_name))
            precisions[input_name] = input_precision
            return input_precision

        constant_inputs = {}
        non_constant_inputs_mapping = {}
        config_non_constant_inputs = []
        layouts = {}
        precisions = {}
        image_info_inputs = []
        lstm_inputs = []

        for input_ in inputs_entry:
            name = input_['name']
            if name not in self.network_inputs:
                raise ConfigError('network does not contain input "{}"'.format(name))

            if input_['type'] == 'IMAGE_INFO':
                image_info_inputs.append(name)
                get_layer_precision(input_, name)
                continue

            if input_['type'] == 'LSTM_INPUT':
                lstm_inputs.append(name)
                get_layer_precision(input_, name)
                continue

            value = input_.get('value')

            if input_['type'] == 'CONST_INPUT':
                if isinstance(value, list):
                    value = np.array(value)
                    precision = get_layer_precision(input_, name)
                    value = value.astype(precision) if precision is not None else value
                constant_inputs[name] = value
            else:
                config_non_constant_inputs.append(name)
                if value is not None:
                    value = re.compile(value) if not isinstance(value, int) else value
                    non_constant_inputs_mapping[name] = value
                layout = input_.get('layout', default_layout)
                layouts[name] = LAYER_LAYOUT_TO_IMAGE_LAYOUT[layout]
                get_layer_precision(input_, name)

        all_config_inputs = config_non_constant_inputs + list(constant_inputs.keys()) + image_info_inputs + lstm_inputs
        not_config_inputs = [input_layer for input_layer in self.network_inputs if input_layer not in all_config_inputs]
        if config_non_constant_inputs and not_config_inputs:
            raise ConfigError('input value for {} are not presented in config.'.format(','.join(not_config_inputs)))
        non_constant_inputs = not_config_inputs + config_non_constant_inputs

        return (
            constant_inputs,
            non_constant_inputs,
            non_constant_inputs_mapping or None,
            image_info_inputs,
            lstm_inputs,
            layouts,
            precisions
        )

    def _transform_batch(self, batch_data, meta):
        def calculate_num_splits(layers_data, batch_size):
            max_split_num = 1
            for _, data in layers_data.items():
                total_tiles_num = 0
                for tiles in data:
                    total_tiles_num += len(tiles)

                offset = 0 if total_tiles_num % batch_size == 0 else 1
                splits_for_layer = (total_tiles_num // batch_size) + offset
                if max_split_num < splits_for_layer:
                    max_split_num = splits_for_layer

            return max_split_num

        def separate_data(data, num_splits):
            grouped_data = [[] for _ in range(num_splits)]
            for data_part in data:
                for split_id, data_split in enumerate(data_part):
                    grouped_data[split_id % num_splits].append(data_split)
            return grouped_data

        batch_size = len(meta)
        if meta[0].get('multi_infer', False):
            num_splits = calculate_num_splits(batch_data, batch_size)
            infers_data = [{} for _ in range(num_splits)]
            for layer_name, layer_data in batch_data.items():
                batch_for_all_infers = separate_data(layer_data, num_splits)
                for infer_id, on_infer_batch in enumerate(batch_for_all_infers):
                    infers_data[infer_id][layer_name] = self.input_transform_func(
                        on_infer_batch, layer_name,
                        self.layouts_mapping.get(layer_name, LAYER_LAYOUT_TO_IMAGE_LAYOUT[self.default_layout]),
                        self.precision_mapping.get(layer_name)
                    )
            return infers_data

        for layer_name, layer_data in batch_data.items():
            batch_data[layer_name] = self.input_transform_func(
                layer_data, layer_name,
                self.layouts_mapping.get(layer_name, LAYER_LAYOUT_TO_IMAGE_LAYOUT[self.default_layout]),
                self.precision_mapping.get(layer_name)
            )

        return [batch_data]

    def release(self):
        del self.network_inputs
