# Copyright (c) 2021-2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib

from pathlib import Path

class DeserializationError(Exception):
    def __init__(self, problem, contexts=()):
        super().__init__(': '.join(contexts + (problem,)))
        self.problem = problem
        self.contexts = contexts

@contextlib.contextmanager
def deserialization_context(context):
    try:
        yield None
    except DeserializationError as exc:
        raise DeserializationError(exc.problem, (context,) + exc.contexts) from exc

def validate_string(context, value):
    if not isinstance(value, str):
        raise DeserializationError('{}: expected a string, got {!r}'.format(context, value))
    return value

def validate_string_enum(context, value, known_values):
    str_value = validate_string(context, value)
    if str_value not in known_values:
        raise DeserializationError('{}: expected one of {!r}, got {!r}'.format(context, known_values, value))
    return str_value

def validate_relative_path(context, value):
    path = Path(validate_string(context, value))

    if path.anchor or '..' in path.parts:
        raise DeserializationError('{}: disallowed absolute path or parent traversal'.format(context))

    return path

def validate_nonnegative_int(context, value):
    if not isinstance(value, int) or value < 0:
        raise DeserializationError(
            '{}: expected a non-negative integer, got {!r}'.format(context, value))
    return value

def validate_nonnegative_float(context, value):
    if not isinstance(value, float) or value < 0:
        raise DeserializationError(
            '{}: expected a non-negative integer, got {!r}'.format(context, value))
    return value

def validate_list(context, value):
    if not isinstance(value, list):
        raise DeserializationError(
            '{}: expected a list, got {!r}'.format(context, value))
    return value
