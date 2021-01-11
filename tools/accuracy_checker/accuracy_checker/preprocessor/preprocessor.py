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

from ..config import ConfigValidator, StringField, ConfigError
from ..dependency import ClassProvider, UnregisteredProviderException
from ..utils import get_parameter_value_from_config


MULTI_INFER_PREPROCESSORS = ['tiling', 'normalize3d', 'image_pyramid', 'clip_audio']


class Preprocessor(ClassProvider):
    __provider_type__ = 'preprocessor'

    def __init__(self, config, name=None):
        self.config = config
        self.name = name
        self.input_shapes = None

        self.validate_config(config)
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

    @classmethod
    def validate_config(cls, config, fetch_only=False, uri_prefix=''):
        errors = []
        if cls.__name__ == Preprocessor.__name__:
            processing_provider = config.get('type')
            if not processing_provider:
                error = ConfigError('type is not found', config, uri_prefix or 'preprocessing')
                if not fetch_only:
                    raise error
                errors.append(error)
                return errors
            try:
                preprocessor_cls = cls.resolve(processing_provider)
            except UnregisteredProviderException as exception:
                if not fetch_only:
                    raise exception
                errors.append(
                    ConfigError(
                        "preprocessor {} unregistered".format(processing_provider), config,
                        uri_prefix or 'preprocessing')
                )
                return errors
            errors.extend(preprocessor_cls.validate_config(config, fetch_only=fetch_only, uri_prefix=uri_prefix))
            return errors

        preprocessor_uri = '{}.{}'.format(uri_prefix or 'preprocessing', cls.__provider__)
        return ConfigValidator(
            preprocessor_uri, on_extra_argument=ConfigValidator.ERROR_ON_EXTRA_ARGUMENT, fields=cls.parameters()
        ).validate(config, fetch_only=fetch_only)

    def set_input_shape(self, input_shape):
        pass
