"""
Copyright (c) 2018-2024 Intel Corporation

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
from collections import defaultdict
import numpy as np

from ..config import ConfigError
from ..utils import extract_image_representations
from ..data_readers import (
    MultiFramesInputIdentifier,
    KaldiFrameIdentifier,
    KaldiMatrixIdentifier,
    ParametricImageIdentifier,
    AnnotationDataIdentifier
)

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
    'NHWDC': [0, 2, 3, 1, 4],
    'NHWCD': [0, 2, 3, 4, 1],
    'NC': [0, 1],
    'CN': [1, 0],
    'CNH': [1, 0, 2],
    'NCH': [0, 1, 2],
    'NSH': [0, 1, 2],
    'N': [0]
}

DIM_IDS_TO_LAYOUT = {
    (0, 3, 1, 2): 'NCHW',
    (0, 1, 2, 3): 'NHWC',
    (0, 4, 1, 2, 3): 'NCDHW',
    (0, 1, 4, 2, 3): 'NDHWC',
    (0, 1, 4, 3, 2): 'NDCWH',
    (0, 1): 'NC',
    (1, 0): 'CN',
    (1, 0, 2): 'CNH',
    (0, 1, 2): 'NCH'
}

PRECISION_TO_DTYPE = {
    'FP32': np.float32,  # float
    'FP16': np.float16,  # signed short
    'U8': np.uint8,  # unsigned char
    'U16': np.uint16,  # unsigned short
    'I8': np.int8,  # signed char
    'I16': np.int16,  # signed short
    'I32': np.int32,  # signed int
    'I64': int,  # signed long int
    'STR': str,  # string'
    'BOOL': bool,
    'f32': np.float32,
    'i32': np.int32,
    'i64': np.int64,
    'fp16': np.float16,
    'f16': np.float16,
    'i16': np.int16,
    'u16': np.uint16,
    'i8': np.int8,
    'u8': np.uint8,
    'boolean': np.uint8
}

INPUT_TYPES_WITHOUT_VALUE = ['IMAGE_INFO', 'ORIG_IMAGE_INFO', 'PROCESSED_IMAGE_INFO', 'IGNORE_INPUT', 'LSTM_INPUT',
                             'SCALE_FACTOR']


def match_by_regex(data, identifiers, input_regex, templates):
    if isinstance(identifiers, AnnotationDataIdentifier):
        identifiers = identifiers.data_id
        if not isinstance(identifiers, list):
            data = [data]
    if not isinstance(identifiers, list):
        identifiers = [identifiers]
    input_data = None
    shape_template = None
    for identifier, data_value, template in zip(identifiers, data, templates):
        if input_regex.match(identifier):
            input_data = data_value
            shape_template = template
            break
    return input_data, shape_template


class InputFeeder:
    def __init__(
            self, inputs_config, network_inputs, shape_checker, prepare_input_data=None, default_layout='NCHW',
            dummy=False, input_precisions_list=None, input_layouts=None, network_input_mapping=None
    ):
        def fit_to_input(data, input_layer_name, layout, precision, template=None):
            layout_used = False
            if np.ndim(data) == 4:
                layout_used = True
                data = np.transpose(data, layout)
            else:
                data = np.array(data)
            if template:
                if len(template) != data.ndim:
                    template = [1] * (data.ndim - len(template)) + list(template)
                if layout_used:
                    new_template = [template[l_dim] for l_dim in layout]
                    template = new_template
                return data.astype(precision) if precision else data, template
            return data.astype(precision) if precision else data

        self.shape_checker = shape_checker
        self.input_transform_func = prepare_input_data or fit_to_input
        self.network_inputs = network_inputs or []
        self.network_input_mapping = network_input_mapping or {
            input_name: input_name for input_name in self.network_inputs}
        self.default_layout = default_layout
        self.dummy = dummy
        self.ordered_inputs = False
        self.configure(inputs_config, input_precisions_list, input_layouts)

    def configure(self, inputs_config, precisions_list, layouts):
        if not self.dummy:
            parsing_results = self._parse_inputs_config(inputs_config, self.default_layout, precisions_list, layouts)
            self.const_inputs, self.non_constant_inputs, self.inputs_mapping = parsing_results[:3]
            (self.image_info_inputs, self.orig_image_info_inputs, self.processed_image_info_inputs,
             self.scale_factor_inputs) = parsing_results[3:7]
            self.lstm_inputs = parsing_results[7]
            self.ignore_inputs, self.layouts_mapping, self.precision_mapping, self.inputs_config = parsing_results[8:]
            if not self.non_constant_inputs:
                raise ConfigError('Network should contain at least one layer for setting variable data.')

    def _fill_image_info_inputs(self, data_representation_batch):
        def prepare_image_info(image_sizes_batch, input_name, preprocessed_input_info=False):
            image_info = []
            input_shape = self.shape_checker(input_name)
            for image_size in image_sizes_batch:
                if np.isscalar(image_size):
                    image_info.append(image_size)
                    continue

                height, width = image_size[:2]
                image_info_ = [height, width]
                info_size = input_shape[-1]
                if info_size == 3:
                    image_info_ += ([1] if not preprocessed_input_info else [image_size[2]])
                if info_size == 6:
                    image_info_ += [0, 0, 0, 0]
                image_info.append(image_info_)

            return image_info

        def prepare_scale_factor(image_meta):
            if 'scale_x' in image_meta[0]:
                return [[meta['scale_y'], meta['scale_x']] for meta in image_meta]
            return [[1, 1] for _ in image_meta]

        data_batch, meta_batch = extract_image_representations(data_representation_batch, meta_only=False)
        image_infos = {}
        if self.processed_image_info_inputs:
            image_info_data = [np.shape(data) for data in data_batch]
            image_infos = {
                processed_image_info_input:
                    prepare_image_info(image_info_data, processed_image_info_input, False)
                for processed_image_info_input in self.processed_image_info_inputs
            }
            return image_infos
        im_info_resolved = False
        if 'image_info' in meta_batch[0]:
            image_info_data = [meta['image_info'] for meta in meta_batch]
            image_infos = {
                image_info_input:
                    prepare_image_info(image_info_data, image_info_input, True)
                for image_info_input in self.image_info_inputs
            }
            im_info_resolved = True
        if im_info_resolved and not self.orig_image_info_inputs and not self.scale_factor_inputs:
            return image_infos
        if self.scale_factor_inputs:
            scale_info = prepare_scale_factor(meta_batch)
            update_image_infos = {input_name: scale_info for input_name in self.scale_factor_inputs}
            image_infos.update(update_image_infos)
        image_sizes = [meta['image_size'] for meta in meta_batch]
        image_infos.update({image_info_input: prepare_image_info(image_sizes, image_info_input)
                            for image_info_input in self.orig_image_info_inputs})
        if not im_info_resolved:
            image_infos.update(
                {image_info_input: prepare_image_info(image_sizes, image_info_input)
                 for image_info_input in self.image_info_inputs})

        return image_infos

    def fill_non_constant_inputs(self, data_representation_batch):
        def match_regex(data, identifiers, input_regex):
            if not isinstance(identifiers, list):
                identifiers = [identifiers]
            input_data = None
            for identifier, data_value in zip(identifiers, data):
                if input_regex.match(identifier):
                    input_data = data_value
                    break
            return input_data

        filled_inputs = {}
        check_regex = True
        if (
            self.image_info_inputs or self.orig_image_info_inputs or
            self.processed_image_info_inputs or self.scale_factor_inputs
        ):
            image_info_inputs = self._fill_image_info_inputs(data_representation_batch)
            filled_inputs = {**image_info_inputs}
        for idx, input_layer in enumerate(self.non_constant_inputs):
            input_batch = []
            input_regex = (self.inputs_mapping or {}).get(input_layer)
            for data_representation in data_representation_batch:
                identifiers = data_representation.identifier
                data = data_representation.data
                if isinstance(identifiers, AnnotationDataIdentifier):
                    identifiers = identifiers.data_id

                if isinstance(identifiers, ParametricImageIdentifier):
                    input_batch.append(data[idx] if input_regex is None else data[input_regex])
                    continue

                if not isinstance(identifiers, list) and input_regex is None:
                    input_data = data
                    input_batch.append(input_data)
                    continue
                if (
                        isinstance(identifiers, list) and
                        isinstance(identifiers[0], (KaldiFrameIdentifier, KaldiMatrixIdentifier))
                ):
                    check_regex = False
                    self.ordered_inputs = True

                if input_regex is None and check_regex:
                    raise ConfigError('Impossible to choose correct data for layer {}.'
                                      'Please provide regular expression for matching in config.'.format(input_layer))

                if isinstance(identifiers, MultiFramesInputIdentifier):
                    input_id_order = {
                        input_index: frame_id for frame_id, input_index in enumerate(identifiers.input_id)
                    }
                    input_data = data[input_id_order[input_regex]]
                else:
                    data = [data] if np.isscalar(identifiers) else data
                    if self.ordered_inputs:
                        assert idx < len(identifiers), 'number input layers and data is not matched'
                        input_batch.append(data[idx])
                        continue
                    input_data = match_regex(data, identifiers, input_regex)

                if input_data is None:
                    raise ConfigError('Suitable data for filling layer {} not found'.format(input_layer))
                input_batch.append(input_data)

            filled_inputs[input_layer] = input_batch

        return self._transform_batch(
            filled_inputs, extract_image_representations(data_representation_batch, meta_only=True)
        )[0]

    def fill_non_constant_inputs_with_template(self, data_representation_batch, template):
        filled_inputs = {}
        filled_template = {}
        check_regex = True
        if (
            self.image_info_inputs or self.orig_image_info_inputs or
            self.processed_image_info_inputs or self.scale_factor_inputs
        ):
            image_info_inputs = self._fill_image_info_inputs(data_representation_batch)
            filled_inputs = {**image_info_inputs}
        for idx, input_layer in enumerate(self.non_constant_inputs):
            input_batch = []
            input_regex = (self.inputs_mapping or {}).get(input_layer)
            shape_template = template[idx]
            for data_representation in data_representation_batch:
                identifiers = data_representation.identifier
                data = data_representation.data
                if isinstance(identifiers, ParametricImageIdentifier):
                    input_batch.append(data[idx])
                    continue

                if not isinstance(identifiers, list) and input_regex is None:
                    input_data = data
                    input_batch.append(input_data)
                    continue
                if (
                        isinstance(identifiers, list) and
                        isinstance(identifiers[0], (KaldiFrameIdentifier, KaldiMatrixIdentifier))
                ):
                    check_regex = False
                    self.ordered_inputs = True

                if input_regex is None and check_regex:
                    raise ConfigError('Impossible to choose correct data for layer {}.'
                                      'Please provide regular expression for matching in config.'.format(input_layer))

                if isinstance(identifiers, MultiFramesInputIdentifier):
                    input_id_order = {
                        input_index: frame_id for frame_id, input_index in enumerate(identifiers.input_id)
                    }
                    input_data = data[input_id_order[input_regex]]
                    shape_template = template[input_id_order[input_regex]]
                else:
                    data = [data] if np.isscalar(identifiers) else data
                    if self.ordered_inputs:
                        assert idx < len(identifiers), 'number input layers and data is not matched'
                        input_batch.append(data[idx])
                        continue
                    input_data, shape_template = match_by_regex(data, identifiers, input_regex, template)

                if input_data is None:
                    raise ConfigError('Suitable data for filling layer {} not found'.format(input_layer))
                input_batch.append(input_data)

            filled_inputs[input_layer] = input_batch
            filled_template[input_layer] = shape_template

        return self._transform_batch(
            filled_inputs, extract_image_representations(data_representation_batch, meta_only=True), filled_template
        )

    def fill_inputs(self, data_representation_batch):
        if self.dummy:
            return []
        inputs = self.fill_non_constant_inputs(data_representation_batch)
        for infer_inputs in inputs:
            infer_inputs.update(self.const_inputs)
        return inputs

    def fill_inputs_with_template(self, data_representation_batch, template=None):
        if self.dummy:
            return [], None
        if template is None:
            return self.fill_inputs(data_representation_batch), None

        inputs, templates = self.fill_non_constant_inputs_with_template(data_representation_batch, template)
        for infer_inputs in inputs:
            infer_inputs.update(self.const_inputs)
        return inputs, templates

    def _parse_inputs_config(self, inputs_entry, default_layout='NCHW', precisions_list=None, layouts=None):
        precision_info = self.validate_input_precision(precisions_list)
        layouts_info = self.validate_input_layouts(layouts, self.network_inputs)
        constant_inputs = {}
        non_constant_inputs_mapping = {}
        config_non_constant_inputs = []
        layouts = {}
        precisions = {}
        image_info_inputs = []
        orig_image_info_inputs = []
        processed_image_info_inputs = []
        lstm_inputs = []
        ignore_inputs = []
        scale_factor_inputs = []

        for input_ in inputs_entry:
            name = input_['name']
            normalized_name = self.network_input_mapping.get(name)
            if normalized_name is None:
                raise ConfigError('network does not contain input "{}"'.format(name))
            input_["name"] = normalized_name
            if input_['type'] in INPUT_TYPES_WITHOUT_VALUE:
                self._configure_inputs_without_value(
                    input_, image_info_inputs, orig_image_info_inputs, processed_image_info_inputs, scale_factor_inputs,
                    lstm_inputs, ignore_inputs, precision_info, precisions)
                continue

            value = input_.get('value')

            if input_['type'] == 'CONST_INPUT':
                precision = self.get_layer_precision(input_, normalized_name, precision_info, precisions) or np.float32
                if isinstance(value, list):
                    value = np.array(value, dtype=precision)
                if isinstance(value, (int, float)) and 'shape' in input_:
                    value = np.full(input_['shape'], value, dtype=precision)
                constant_inputs[normalized_name] = self.input_transform_func(value, normalized_name, None, precision)
            else:
                config_non_constant_inputs.append(normalized_name)
                if value is not None:
                    value = re.compile(value) if not isinstance(value, int) else value
                    non_constant_inputs_mapping[normalized_name] = value
                layout = layouts_info.get(normalized_name, input_.get('layout', default_layout))
                if normalized_name in layouts_info:
                    input_['layout'] = layout
                if layout in LAYER_LAYOUT_TO_IMAGE_LAYOUT:
                    layouts[normalized_name] = LAYER_LAYOUT_TO_IMAGE_LAYOUT[layout]
                self.get_layer_precision(input_, normalized_name, precision_info, precisions)

        all_config_inputs = (
            config_non_constant_inputs + list(constant_inputs.keys()) +
            image_info_inputs + lstm_inputs + orig_image_info_inputs + ignore_inputs + scale_factor_inputs +
            processed_image_info_inputs
        )
        not_config_inputs = [input_layer for input_layer in self.network_inputs if input_layer not in all_config_inputs]
        if config_non_constant_inputs and not_config_inputs:
            raise ConfigError('input value for {} are not presented in config.'.format(','.join(not_config_inputs)))
        non_constant_inputs = not_config_inputs + config_non_constant_inputs
        if not_config_inputs and (precision_info or isinstance(precision_info, defaultdict)) or layouts_info:
            inputs_entry = self.provide_input_config_for_not_config(
                inputs_entry, precision_info, not_config_inputs, precisions, layouts_info, layouts
            )

        return (
            constant_inputs,
            non_constant_inputs,
            non_constant_inputs_mapping or None,
            image_info_inputs,
            orig_image_info_inputs,
            processed_image_info_inputs,
            scale_factor_inputs,
            lstm_inputs,
            ignore_inputs,
            layouts,
            precisions,
            inputs_entry
        )

    def _configure_inputs_without_value(
            self, input_config, image_info_inputs,
            orig_image_info_inputs, processed_image_info_inputs, scale_factor_inputs, lstm_inputs, ignore_inputs,
            precision_info, precisions):
        name = input_config['name']
        if input_config['type'] == 'IMAGE_INFO':
            image_info_inputs.append(name)
            self.get_layer_precision(input_config, name, precision_info, precisions)

        if input_config['type'] == 'ORIG_IMAGE_INFO':
            orig_image_info_inputs.append(name)
            self.get_layer_precision(input_config, name, precision_info, precisions)

        if input_config['type'] == 'PROCESSED_IMAGE_INFO':
            processed_image_info_inputs.append(name)
            self.get_layer_precision(input_config, name, precision_info, precisions)

        if input_config['type'] == 'SCALE_FACTOR':
            scale_factor_inputs.append(name)
            self.get_layer_precision(input_config, name, precision_info, precisions)

        if input_config['type'] == 'LSTM_INPUT':
            lstm_inputs.append(name)
            self.get_layer_precision(input_config, name, precision_info, precisions)

        if input_config['type'] == 'IGNORE_INPUT':
            ignore_inputs.append(name)

    def _transform_batch(self, batch_data, meta, template=None):
        def calculate_num_splits(layers_data, batch_size):
            max_split_num = 1
            for _, data in layers_data.items():
                total_tiles_num = 0
                for tiles in data:
                    total_tiles_num += len(tiles)

                offset = 0 if total_tiles_num % batch_size == 0 else 1
                splits_for_layer = (total_tiles_num // batch_size) + offset
                max_split_num = max(max_split_num, splits_for_layer)

            return max_split_num

        def separate_data(data, num_splits):
            grouped_data = [[] for _ in range(num_splits)]
            for data_part in data:
                for split_id, data_split in enumerate(data_part):
                    grouped_data[split_id % num_splits].append(data_split)
            return grouped_data

        batch_size = len(meta)
        template_for_shapes = {}
        if meta[0].get('multi_infer', False):
            num_splits = calculate_num_splits(batch_data, batch_size)
            infers_data = [{} for _ in range(num_splits)]
            for layer_name, layer_data in batch_data.items():
                tmpl = template.get(layer_name) if template is not None else None
                batch_for_all_infers = separate_data(layer_data, num_splits)
                for infer_id, on_infer_batch in enumerate(batch_for_all_infers):
                    res = self.input_transform_func(
                        on_infer_batch, layer_name,
                        self.layouts_mapping.get(layer_name, LAYER_LAYOUT_TO_IMAGE_LAYOUT[self.default_layout]),
                        self.precision_mapping.get(layer_name),
                        tmpl
                    )
                    infers_data[infer_id][layer_name] = res[0] if isinstance(res, tuple) else res
                    if isinstance(res, tuple):
                        template_for_shapes[layer_name] = res[1]
            return infers_data, template_for_shapes

        for layer_name, layer_data in batch_data.items():
            layout = self.layouts_mapping.get(layer_name)
            if 'data_layout' in meta[0]:
                data_layout = LAYER_LAYOUT_TO_IMAGE_LAYOUT.get(meta[0]['data_layout'])
                if layout is None and len(self.default_layout) == len(data_layout):
                    layout = LAYER_LAYOUT_TO_IMAGE_LAYOUT[self.default_layout]
                if layout is not None and data_layout == layout:
                    layout = []
            if layout is None:
                layout = LAYER_LAYOUT_TO_IMAGE_LAYOUT[self.default_layout]
            layer_data_preprocessed = self.input_transform_func(
                layer_data, layer_name,
                layout,
                self.precision_mapping.get(layer_name), template
            )
            if isinstance(layer_data_preprocessed, tuple):
                layer_template = layer_data_preprocessed[1]
                if layer_template is not None:
                    template_for_shapes = layer_template
                layer_data_preprocessed = layer_data_preprocessed[0]
            batch_data[layer_name] = layer_data_preprocessed

        return [batch_data], template_for_shapes

    @staticmethod
    def validate_input_layouts(parameter_string, input_names):
        # Parse parameter string like "input0[value0],input1[value1]" or "[value]" (applied to all inputs)
        return_value = {}
        if parameter_string:
            matches = re.findall(r'(.*?)\[(.*?)\],?', parameter_string)
            if matches:
                for match in matches:
                    input_name, value = match
                    if input_name != '':
                        return_value[input_name] = value
                    else:
                        return_value = {k: value for k in input_names}
                        break
            else:
                raise ConfigError(f"Can't parse input parameter: {parameter_string}")
        return return_value

    def validate_input_precision(self, precisions_list):
        if not precisions_list:
            return {}
        if len(precisions_list) == 1 and len(precisions_list[0].rsplit(':', 1)) == 1:
            return defaultdict(lambda: precisions_list[0])
        precision_dict = {}
        for input_c in precisions_list:
            precision_for_layer = input_c.rsplit(':', 1)
            if len(precision_for_layer) == 1:
                raise ConfigError(
                    'invalid value for input precision {}. Please specify <input_name>:<precision>'.format(input_c)
                )
            layer_name, precision_ = precision_for_layer
            input_name = self.network_input_mapping.get(layer_name)
            if input_name is None:
                raise ConfigError("precision specified for unknown layer: {}".format(layer_name))
            precision_dict[input_name] = precision_
        return precision_dict

    @staticmethod
    def get_layer_precision(input_config, input_name, precision_info, precisions):
        precision = (
            precision_info.get(input_name) if not isinstance(precision_info, defaultdict)
            else precision_info[input_name]
        )
        if precision is not None:
            input_config['precision'] = precision
        if 'precision' not in input_config:
            return None
        input_precision = PRECISION_TO_DTYPE.get(input_config['precision'].upper())
        if input_precision is None:
            raise ConfigError("unsupported precision {} for layer {}".format(input_config['precision'], input_name))
        precisions[input_name] = input_precision
        return input_precision

    def provide_input_config_for_not_config(
        self, inputs_entry, precision_info, not_config_inputs, precisions, layouts, layouts_mapping):
        for input_name in not_config_inputs:
            input_config = {'name': input_name, 'type': 'INPUT'}
            precision = self.get_layer_precision(input_config, input_name, precision_info, precisions)
            layout = layouts.get(input_name)
            if layout in LAYER_LAYOUT_TO_IMAGE_LAYOUT:
                layouts_mapping[input_name] = LAYER_LAYOUT_TO_IMAGE_LAYOUT[layout]
            input_config['layout'] = layout
            if precision is not None or layout is not None:
                inputs_entry.append(input_config)
        return inputs_entry

    def update_layout_configuration(self, layout_mapping, override=False):
        for layer_name, layout in layout_mapping.items():
            if layer_name in self.layouts_mapping:
                if not override:
                    continue
                if layout in LAYER_LAYOUT_TO_IMAGE_LAYOUT:
                    self.layouts_mapping[layer_name] = LAYER_LAYOUT_TO_IMAGE_LAYOUT[layout]
                else:
                    del self.layouts_mapping[layer_name]
            elif layout in LAYER_LAYOUT_TO_IMAGE_LAYOUT:
                self.layouts_mapping[layer_name] = LAYER_LAYOUT_TO_IMAGE_LAYOUT[layout]

    def release(self):
        del self.network_inputs
