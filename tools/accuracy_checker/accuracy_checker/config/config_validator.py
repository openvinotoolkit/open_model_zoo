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

import enum
import math
import re
import warnings
from collections import OrderedDict
from copy import copy
from functools import partial
from pathlib import Path

from ..utils import get_path, cast_to_bool


class ConfigError(ValueError):
    def __init__(self, message="", entry='', field_uri=''):
        self.entry = entry
        self.field_uri = field_uri
        self.message = message
        super().__init__(self.message)


class BaseValidator:
    def __init__(self, on_error=None, additional_validator=None):
        self.on_error = on_error
        self.additional_validator = additional_validator

        self.field_uri = None

    def validate(self, entry, field_uri=None, fetch_only=False):
        field_uri = field_uri or self.field_uri
        errors = []
        if self.additional_validator and not self.additional_validator(entry, field_uri):
            if not fetch_only:
                self.raise_error(entry, field_uri)
            else:
                errors.append(self.build_error(entry, field_uri))
        return errors

    def raise_error(self, value, field_uri, reason=None, override_message=False):
        if self.on_error:
            self.on_error(value, field_uri, reason)
        error = self.build_error(value, field_uri, reason, override_message)
        raise error

    @staticmethod
    def build_error(value, field_uri, reason=None, override_message=False):
        error_message = 'Invalid value "{value}" for {field_uri}'.format(value=value, field_uri=field_uri)
        if reason:
            if not override_message:
                error_message = '{error_message}: {reason}'.format(error_message=error_message, reason=reason)
            else:
                error_message = reason

        return ConfigError(error_message, value, field_uri)


class _ExtraArgumentBehaviour(enum.Enum):
    WARN = 'warn'
    IGNORE = 'ignore'
    ERROR = 'error'


def _is_dict_like(entry):
    return hasattr(entry, '__iter__') and hasattr(entry, '__getitem__')


class ConfigValidator(BaseValidator):
    WARN_ON_EXTRA_ARGUMENT = _ExtraArgumentBehaviour.WARN
    ERROR_ON_EXTRA_ARGUMENT = _ExtraArgumentBehaviour.ERROR
    IGNORE_ON_EXTRA_ARGUMENT = _ExtraArgumentBehaviour.IGNORE
    acceptable_unknown_options = ['connector']

    def __init__(self, config_uri, on_extra_argument=WARN_ON_EXTRA_ARGUMENT, fields=None, **kwargs):
        super().__init__(**kwargs)
        self.on_extra_argument = on_extra_argument

        self.fields = OrderedDict()
        self.field_uri = config_uri

        if fields:
            for name in fields.keys():
                self.fields[name] = fields[name]
                if fields[name].field_uri is None:
                    fields[name].field_uri = "{}.{}".format(config_uri, name)
        else:
            for name in dir(self):
                value = getattr(self, name)
                if not isinstance(value, BaseField):
                    continue

                field_copy = copy(value)
                field_copy.field_uri = "{}.{}".format(config_uri, name)
                self.fields[name] = field_copy

    def validate(self, entry, field_uri=None, fetch_only=False):
        error_stack = super().validate(entry, field_uri, fetch_only=fetch_only)
        field_uri = field_uri or self.field_uri
        if not _is_dict_like(entry):
            error = ConfigError("{} is expected to be dict-like".format(field_uri))
            if not fetch_only:
                raise error
            error_stack.append(error)

        extra_arguments = []
        for key in entry:
            if key in self.acceptable_unknown_options:
                continue
            if key not in self.fields:
                extra_arguments.append(key)
                continue

            error_stack.extend(self.fields[key].validate(entry[key], fetch_only=fetch_only))

        required_fields = set(name for name, value in self.fields.items() if value.required())
        missing_arguments = required_fields.difference(entry)

        if missing_arguments:
            arguments = ', '.join(map(str, missing_arguments))
            missing_error = "Invalid config for {}: missing required fields: {}".format(field_uri, arguments)
            if not fetch_only:
                self.raise_error(
                    entry, field_uri, missing_error, override_message=True
                )
            else:
                error_stack.append(self.build_error(entry, field_uri, missing_error, override_message=True))

        if extra_arguments:
            error_stack.extend(self._extra_args_error_handle(entry, field_uri, extra_arguments, fetch_only))

        return error_stack

    def _extra_args_error_handle(self, entry, field_uri, extra_arguments, fetch_only=False):
        unknown_options_error = "specifies unknown options: {}".format(extra_arguments)
        message = "{} {}".format(field_uri, unknown_options_error)

        if self.on_extra_argument == _ExtraArgumentBehaviour.WARN:
            warnings.warn(message)
        if self.on_extra_argument == _ExtraArgumentBehaviour.ERROR:
            if not fetch_only:
                self.raise_error(entry, field_uri, message)
            else:
                return [self.build_error(entry, field_uri, message, override_message=True)]
        return []

    @property
    def known_fields(self):
        return set(self.fields)


class BaseField(BaseValidator):
    def __init__(self, optional=False, description=None, default=None, **kwargs):
        super().__init__(**kwargs)
        self.optional = optional
        self.description = description
        self.default = default

    def validate(self, entry, field_uri=None, fetch_only=False):
        error_stack = super().validate(entry, field_uri, fetch_only)
        field_uri = field_uri or self.field_uri
        empty_error = "{} is not allowed to be None"
        if self.required() and entry is None:
            if not fetch_only:
                self.raise_error(entry, field_uri, empty_error.format(field_uri))
            else:
                error_stack.append(self.build_error(entry, field_uri, empty_error.format(field_uri)))
        return error_stack

    @property
    def type(self):
        return None

    def required(self):
        return not self.optional and self.default is None

    def parameters(self):
        parameters_dict = {}
        for key, _ in self.__dict__.items():
            if not key.startswith('_') and hasattr(self, key) and not hasattr(BaseValidator(), key):
                if isinstance(self.__dict__[key], BaseField):
                    parameters_dict[key] = self.__dict__[key].parameters()
                else:
                    parameters_dict[key] = self.__dict__[key]
            parameters_dict['type'] = type((self.type or str)()).__name__

        return parameters_dict


class StringField(BaseField):
    def __init__(self, choices=None, regex=None, case_sensitive=False, allow_own_choice=False, **kwargs):
        super().__init__(**kwargs)
        self.choices = choices if case_sensitive or not choices else list(map(str.lower, choices))
        self.allow_own_choice = allow_own_choice
        self.case_sensitive = case_sensitive
        self.set_regex(regex)

    def set_regex(self, regex):
        if regex is None:
            self._regex = regex
        self._regex = re.compile(regex, flags=re.IGNORECASE if not self.case_sensitive else 0) if regex else None

    def validate(self, entry, field_uri=None, fetch_only=False):
        error_stack = super().validate(entry, field_uri)
        if entry is None:
            return error_stack

        field_uri = field_uri or self.field_uri
        source_entry = entry

        if not isinstance(entry, str):
            type_error = "{} is expected to be str"
            if not fetch_only:
                self.raise_error(entry, field_uri, type_error.format(source_entry))
            else:
                error_stack.append(self.build_error(entry, field_uri, type_error.format(source_entry)))
                return error_stack

        if not self.case_sensitive:
            entry = entry.lower()

        if self.choices and entry not in self.choices and not self.allow_own_choice:
            reason = "unsupported option, expected one of: {}".format(', '.join(map(str, self.choices)))
            if not fetch_only:
                self.raise_error(source_entry, field_uri, reason)
            else:
                error_stack.append(self.build_error(source_entry, field_uri, reason))

        if self._regex and not self._regex.match(entry):
            regex_reason = 'value does not matched by regex'
            if not fetch_only:
                self.raise_error(source_entry, field_uri, reason=regex_reason)
            else:
                error_stack.append(self.build_error(source_entry, field_uri, regex_reason))

        return error_stack

    @property
    def type(self):
        return str


class DictField(BaseField):
    def __init__(self, key_type=None, value_type=None, validate_keys=True, validate_values=True, allow_empty=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.validate_keys = validate_keys if key_type else False
        self.validate_values = validate_values if value_type else False
        self.key_type = _get_field_type(key_type)
        self.value_type = _get_field_type(value_type)

        self.allow_empty = allow_empty

    def validate(self, entry, field_uri=None, fetch_only=False):
        error_stack = super().validate(entry, field_uri, fetch_only)
        if entry is None:
            return error_stack

        field_uri = field_uri or self.field_uri
        if not isinstance(entry, dict):
            msg = "{} is expected to be dict".format(field_uri)
            if not fetch_only:
                self.raise_error(entry, field_uri, msg)
            else:
                error_stack.append(self.build_error(entry, field_uri, msg))
                return error_stack

        if not entry and not self.allow_empty:
            empty_msg = "value is empty"
            if not fetch_only:
                self.raise_error(entry, field_uri, empty_msg)
            else:
                error_stack.append(self.build_error(entry, field_uri, empty_msg))
                return error_stack

        for k, v in entry.items():
            if self.validate_keys:
                uri = "{}.keys.{}".format(field_uri, k)
                error_stack.extend(self.key_type.validate(k, uri, fetch_only))

            if self.validate_values:
                uri = "{}.{}".format(field_uri, k)

                error_stack.extend(self.value_type.validate(v, uri, fetch_only))
        return error_stack

    @property
    def type(self):
        return dict


class ListField(BaseField):
    def __init__(self, value_type=None, validate_values=True, allow_empty=True, **kwargs):
        super().__init__(**kwargs)
        self.validate_values = validate_values if value_type else False
        self.value_type = _get_field_type(value_type)
        self.allow_empty = allow_empty

    def validate(self, entry, field_uri=None, fetch_only=False):
        error_stack = super().validate(entry, field_uri, fetch_only)
        if entry is None:
            return error_stack

        if not isinstance(entry, list):
            msg = "{} is expected to be list".format(field_uri)
            if not fetch_only:
                self.raise_error(entry, field_uri, msg)
            else:
                error_stack.append(self.build_error(entry, field_uri, msg))
                return error_stack

        if not entry and not self.allow_empty:
            if not fetch_only:
                self.raise_error(entry, field_uri, "value is empty")
            else:
                error_stack.append(self.build_error(entry, field_uri, "value is empty"))
                return error_stack

        if self.validate_values:
            for i, val in enumerate(entry):
                error_stack.extend(self.value_type.validate(val, "{}[{}]".format(val, i), fetch_only))
        return error_stack

    @property
    def type(self):
        return list


class InputField(BaseField):
    INPUTS_TYPES = ('CONST_INPUT', 'INPUT', 'IMAGE_INFO', 'ORIG_IMAGE_INFO', 'LSTM_INPUT', 'IGNORE_INPUT')
    LAYOUT_TYPES = ('NCHW', 'NHWC', 'NCWH', 'NWHC')
    PRECISIONS = ('FP32', 'FP16', 'U8', 'U16', 'I8', 'I16', 'I32', 'I64')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = StringField(description="Input name.")
        self.input_type = StringField(choices=InputField.INPUTS_TYPES, description="Type name.")
        self.value = BaseField(description="Input value.")
        self.layout = StringField(optional=True, choices=InputField.LAYOUT_TYPES,
                                  description="Layout: " + ', '.join(InputField.LAYOUT_TYPES))
        self.shape = BaseField(optional=True, description="Input shape.")
        self.precision = StringField(optional=True, description='Input precision', choices=InputField.PRECISIONS)

    def validate(self, entry, field_uri=None, fetch_only=False):
        entry['optional'] = entry['type'] not in ['CONST_INPUT', 'LSTM_INPUT']
        return super().validate(entry, field_uri)


class ListInputsField(ListField):
    def __init__(self, **kwargs):
        super().__init__(allow_empty=False, value_type=InputField(description="Input type."), **kwargs)

    def validate(self, entry, field_uri=None, fetch_only=False):
        error_stack = super().validate(entry, field_uri, fetch_only)
        names_set = set()
        for input_layer in entry:
            input_name = input_layer['name']
            if input_name not in names_set:
                names_set.add(input_name)
            else:
                if not fetch_only:
                    self.raise_error(entry, field_uri, '{} repeated name'.format(input_name))
                else:
                    error_stack.append(self.build_error(entry, field_uri, '{} repeated name'.format(input_name)))
        return error_stack


class NumberField(BaseField):
    def __init__(self, value_type=float, min_value=None, max_value=None, allow_inf=False, allow_nan=False, **kwargs):
        super().__init__(**kwargs)
        self._value_type = value_type
        self.min = min_value
        self.max = max_value
        self._allow_inf = allow_inf
        self._allow_nan = allow_nan

    def validate(self, entry, field_uri=None, fetch_only=False):
        error_stack = super().validate(entry, field_uri)
        if entry is None:
            return error_stack

        field_uri = field_uri or self.field_uri
        if self.type != float and isinstance(entry, float):
            if fetch_only:
                error_stack.append(self.build_error(entry, field_uri, "{} is expected to be int".format(field_uri)))
                return error_stack
            self.raise_error(entry, field_uri, "{} is expected to be int".format(field_uri))

        if not isinstance(entry, int) and not isinstance(entry, float):
            if fetch_only:
                error_stack.append(self.build_error(entry, field_uri, "{} is expected to be number".format(field_uri)))
                return error_stack
            self.raise_error(entry, field_uri, "{} is expected to be number".format(field_uri))

        error_stack.extend(self.range_check(entry, field_uri, fetch_only))
        error_stack.extend(self.finite_check(entry, field_uri, fetch_only))

        return error_stack

    def range_check(self, entry, field_uri, fetch_only):
        error_stack = []
        if self.min is not None and entry < self.min:
            reason = "value is less than minimal allowed - {}".format(self.min)
            if not fetch_only:
                self.raise_error(entry, field_uri, reason)
            else:
                error_stack.append(self.build_error(entry, field_uri, reason))
        if self.max is not None and entry > self.max:
            reason = "value is greater than maximal allowed - {}".format(self.max)
            if not fetch_only:
                self.raise_error(entry, field_uri, reason)
            else:
                error_stack.append(self.build_error(entry, fetch_only, reason))
        return error_stack

    def finite_check(self, entry, field_uri, fetch_only):
        error_stack = []
        if math.isinf(entry) and not self._allow_inf:
            reason = "value is infinity"
            if not fetch_only:
                self.raise_error(entry, field_uri, reason)
            else:
                error_stack.append(self.build_error(entry, field_uri, reason))
        if math.isnan(entry) and not self._allow_nan:
            reason = "value is NaN"
            if not fetch_only:
                self.raise_error(entry, field_uri, reason)
            else:
                error_stack.append(self.build_error(entry, field_uri, reason))
        return error_stack

    @property
    def type(self):
        return self._value_type


class PathField(BaseField):
    def __init__(self, is_directory=False, check_exists=True, file_or_directory=False, **kwargs):
        super().__init__(**kwargs)
        self.is_directory = is_directory
        self.check_exists = check_exists
        self.file_or_directory = file_or_directory

    def validate(self, entry, field_uri=None, fetch_only=False):
        error_stack = super().validate(entry, field_uri, fetch_only)
        if entry is None:
            return error_stack

        field_uri = field_uri or self.field_uri
        try:
            get_path(entry, self.is_directory, self.check_exists, self.file_or_directory)
        except TypeError:
            reason = "values is expected to be path-like"
            if not fetch_only:
                self.raise_error(entry, field_uri, reason)
            error_stack.append(self.build_error(entry, field_uri, reason))
        except FileNotFoundError:
            reason = "path does not exist"
            if not fetch_only:
                self.raise_error(entry, field_uri, reason)
            error_stack.append(self.build_error(entry, field_uri, reason))
        except NotADirectoryError:
            reason = "path is not a directory"
            if not fetch_only:
                self.raise_error(entry, field_uri, reason)
            error_stack.append(self.build_error(entry, field_uri, reason))
        except IsADirectoryError:
            reason = "path is a directory, regular file expected"
            if not fetch_only:
                self.raise_error(entry, field_uri, reason)
            error_stack.append(self.build_error(entry, field_uri, reason))

        return error_stack

    @property
    def type(self):
        return Path


class BoolField(BaseField):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def validate(self, entry, field_uri=None, fetch_only=False):
        error_stack = super().validate(entry, field_uri, fetch_only)
        if entry is None:
            return error_stack

        field_uri = field_uri or self.field_uri
        if not isinstance(entry, bool):
            reason = "{} is expected to be bool".format(field_uri)
            if not fetch_only:
                self.raise_error(entry, field_uri, reason)
            else:
                error_stack.append(self.build_error(entry, field_uri, reason))
        return error_stack

    @property
    def type(self):
        return cast_to_bool

    def parameters(self):
        parameters_dict = {}
        for key, _ in self.__dict__.items():
            if not key.startswith('_') and hasattr(self, key) and not hasattr(BaseValidator(), key):
                if isinstance(self.__dict__[key], BaseField):
                    parameters_dict[key] = self.__dict__[key].parameters()
                else:
                    parameters_dict[key] = self.__dict__[key]
            parameters_dict['type'] = type(bool()).__name__
        return parameters_dict


def _get_field_type(key_type):
    if not isinstance(key_type, BaseField):
        type_ = _TYPE_TO_FIELD_CLASS.get(key_type)
        if callable(type_):
            return type_()

    return key_type


_TYPE_TO_FIELD_CLASS = {
    int: partial(NumberField, value_type=int),
    float: partial(NumberField, value_type=float),
    dict: partial(DictField, validate_keys=False, validate_values=False),
    list: partial(ListField, validate_values=False),
    Path: PathField,
    str: StringField,
    bool: BoolField,
}
