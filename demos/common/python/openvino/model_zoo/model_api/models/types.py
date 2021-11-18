"""
 Copyright (C) 2021 Intel Corporation

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

import logging as log

class ConfigError(ValueError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class BaseValue:
    def __init__(self, description="No description available", default_value=None) -> None:
        self.default_value = default_value
        self.description = description

    def update_default_value(self, default_value):
        self.default_value = default_value

    def validate(self, value):
        return []

    def get_value(self, value):
        errors = self.validate(value)
        if len(errors) == 0:
            return value if value else self.default_value

    def build_error():
        pass

    def __str__(self) -> str:
        info = self.description
        if self.default_value:
            info += f"\nThe default value is '{self.default_value}'"
        return info

class NumericalValue(BaseValue):
    def __init__(self, value_type=float, choices=(), min=None, max=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.choices = choices
        self.min = min
        self.max = max
        self.value_type = value_type

    def validate(self, value):
        errors = super().validate(value)
        if not value:
            return errors
        if not isinstance(value, self.value_type):
            errors.append(ValueError('Not type'))
        if len(self.choices):
            if value not in self.choices:
                errors.append(ValueError('Not in choices'))
        if self.min and value < self.min:
            errors.append(ValueError())
        if self.max and value > self.max:
            errors.append( ValueError())
        return errors

    def __str__(self) -> str:
        info = super().__str__()
        info += f"\nAppropriate type is {self.value_type}"
        if self.choices:
            info += f"\nAppropriate values are {self.choices}"
        return info

class StringValue(BaseValue):
    def __init__(self, choices=(), **kwargs):
        super().__init__(**kwargs)
        self.choices = choices
        for choice in self.choices:
            if not isinstance(choice, str):
                raise ValueError("Incorrect option in choice list - {}.". format(choice))

    def validate(self, value):
        errors = super().validate(value)
        if not value:
            return errors
        if not isinstance(value, str):
            errors.append(ValueError("Not string"))
        if value not in self.choices:
            errors.append('Value "{}" not from choices {}'.format(value, self.choices))
        return errors

    def __str__(self) -> str:
        info = super().__str__()
        info += "\nAppropriate type is str"
        if self.choices:
            info += f"\nAppropriate values are {self.choices}"

        return info


class BooleanValue(BaseValue):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def validate(self, value):
        errors = super().validate(value)
        if not value:
            return errors
        if not isinstance(value, bool):
            errors.append(ValueError("Not the boolean"))
        return errors

class ListValue(BaseValue):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def validate(self, value):
        errors = super().validate(value)
        if not value:
            return errors
        if not isinstance(value, (tuple, list)):
            errors.append(ValueError("Not a tuple/list"))   
        return errors 
