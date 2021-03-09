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

from ..config import ConfigValidator, StringField
from ..dependency import ClassProvider
from ..utils import get_parameter_value_from_config


MULTI_INFER_PREPROCESSORS = ['tiling', 'normalize3d', 'image_pyramid', 'clip_audio']


class Preprocessor(ClassProvider):
    __provider_type__ = 'preprocessor'

    def __init__(self, config, name=None):
        self.config = config
        self.name = name
        self.input_shapes = None

        self.validate_config()
        self.configure()

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    def get_value_from_config(self, key):
        return get_parameter_value_from_config(self.config, self.parameters(), key)

    @classmethod
    def parameters(cls):
        return {
            'type': StringField(
                default=cls.__provider__ if hasattr(cls, '__provider__') else None, description="Preprocessor type."
            )
        }

    def process(self, image, annotation_meta=None):
        raise NotImplementedError

    def configure(self):
        pass

    def validate_config(self):
        ConfigValidator(
            self.name, on_extra_argument=ConfigValidator.ERROR_ON_EXTRA_ARGUMENT, fields=self.parameters()
        ).validate(self.config)

    def set_input_shape(self, input_shape):
        pass
