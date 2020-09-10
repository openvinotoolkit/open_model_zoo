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

from argparse import ArgumentParser
from collections import namedtuple

from ..topology_types import GenericTopology
from ..config import ConfigValidator, StringField, PathField, ConfigError
from ..dependency import ClassProvider
from ..utils import format_key, get_parameter_value_from_config

ConverterReturn = namedtuple('ConverterReturn', ['annotations', 'meta', 'content_check_errors'])


class BaseFormatConverter(ClassProvider):
    __provider_type__ = 'converter'
    topology_types = (GenericTopology, )

    @classmethod
    def parameters(cls):
        return {
            'converter': StringField(description="Converter name.")
        }

    @property
    def config_validator(self):
        return ConfigValidator(
            '{}_converter_config'.format(self.get_name()), fields=self.parameters(),
            on_extra_argument=ConfigValidator.ERROR_ON_EXTRA_ARGUMENT
        )

    def __init__(self, config=None):
        self.config = config
        if config:
            self.validate_config()
            self.configure()

    def get_value_from_config(self, key):
        return get_parameter_value_from_config(self.config, self.parameters(), key)

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        """
        Converts specific annotation format to the ResultRepresentation specific for current dataset/task.
        Arguments:
            check_content: bool flag which enable dataset files (e. g. images, gt segmentation masks) existence checking
            progress_callback: callback function for handling conversion progress status

        Returns:
            instance of ConverterReturn, where:
             annotations is list of AnnotationRepresentations for current dataset
             meta is dataset specific attributes e. g. label_map (can be None if dataset does not have specific info)
             content_check_errors: list of error string messages for content check (can be None if check_content=False)
        """
        raise NotImplementedError

    @classmethod
    def get_name(cls):
        return cls.__provider__

    def get_argparser(self):
        parser = ArgumentParser(add_help=False)
        config_validator = self.config_validator
        fields = config_validator.fields
        for field_name, field in fields.items():
            if field_name == 'converter':
                # it is base argument. Main argparser already use it to get argparser from specific converter.
                # Converter argparser should contain only converter specific arguments.
                continue

            kwargs = {'required': not field.optional}
            data_type = field.type
            if data_type is not None:
                kwargs['type'] = data_type

            parser.add_argument(format_key(field_name), **kwargs)

        return parser

    def validate_config(self):
        self.config_validator.validate(self.config)

    def configure(self):
        pass


class FileBasedAnnotationConverter(BaseFormatConverter):
    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'annotation_file': PathField(description="Path to annotation file.")
        })
        return parameters

    def configure(self):
        self.annotation_file = self.get_value_from_config('annotation_file')

    def convert(self, check_content=False, **kwargs):
        pass


class DirectoryBasedAnnotationConverter(BaseFormatConverter):
    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'data_dir': PathField(is_directory=True, description="Path to data directory.")
        })
        return parameters

    def configure(self):
        self.data_dir = self.get_value_from_config('data_dir')

    def convert(self, check_content=False, **kwargs):
        pass


def verify_label_map(label_map):
    valid_label_map = {}
    for class_id, class_name in label_map.items():
        try:
            int_class_id = int(class_id)
            valid_label_map[int_class_id] = class_name
        except ValueError:
            raise ConfigError(
                'class_id {} is invalid. `label_map` should have integer keys.'.format(class_id)
            )
    return valid_label_map
