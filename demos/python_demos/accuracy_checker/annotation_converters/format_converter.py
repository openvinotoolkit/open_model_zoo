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
import inspect
from argparse import ArgumentParser
from itertools import islice

from accuracy_checker.dependency import ClassProvider


class BaseFormatConverter(ClassProvider):
    __provider_type__ = "converter"

    def __init__(self):
        self.image_root = None

    def convert(self, *args, **kwargs):
        """
        Converts specific annotation format to the ResultRepresentation specific for current dataset/task
        Returns:
            annotation: list of ResultRepresentations
            meta: meta-data map for the current dataset
        """
        raise NotImplementedError

    @classmethod
    def get_name(cls):
        return cls.__provider__

    @classmethod
    def get_argparser(cls):
        parser = ArgumentParser(add_help=False)
        signature = inspect.signature(cls.convert)

        for name, param in islice(signature.parameters.items(), 1, None):
            if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                default_value = param.default
                if default_value == inspect.Parameter.empty:
                    if param.annotation == bool:
                        parser.add_argument("--{}".format(name), action='store_true')
                    else:
                        parser.add_argument(name)
                else:
                    parser.add_argument("--{}".format(name), default=default_value, required=False)

        return parser
