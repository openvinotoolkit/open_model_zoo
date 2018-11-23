# pylint: disable=protected-access
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

class ProvidedWrapper:
    def __init__(self, provided):
        self.provided = provided


class UnresolvedDependencyException(ValueError):

    def __init__(self, provider, missing_dependencies) -> None:
        super().__init__()
        self.provider = provider
        self.missing_dependencies = missing_dependencies
        self.message = "Unresolved dependencies ({}) for provider {}".format(", ".join(self.missing_dependencies),
                                                                             self.provider)


def _get_opts(k_):
    """
    Args:
        k_: options object

    Returns:
        args (tuple): positional options
        kwargs (map): keyword arguments
    """
    if isinstance(k_, tuple):
        if len(k_) == 2 and isinstance(k_[-1], dict):
            args, kwargs = k_
        else:
            args = k_
            kwargs = {}
    elif isinstance(k_, dict):
        args, kwargs = (), k_
    else:
        raise ValueError("Options object expected to be either pair of (args, kwargs) or only args/kwargs")
    return args, kwargs


class BaseProvider:
    providers = {}
    __provider_type__ = None
    __provider__ = None

    @classmethod
    def provide(cls, provider, *args, **kwargs):
        root_provider = cls.resolve(provider)
        return root_provider(*args, **kwargs)

    @classmethod
    def resolve(cls, name):
        if name not in cls.providers:
            raise ValueError("Requested provider not registered")
        return cls.providers[name]


class ClassProviderMeta(type):
    def __new__(mcs, name, bases, attrs, **kwargs):
        cls = super().__new__(mcs, name, bases, attrs)
        # do not create container for abstract provider
        if '_is_base_provider' in attrs:
            return cls

        assert issubclass(cls, ClassProvider), "Do not use metaclass directly"
        if '__provider_type__' in attrs:
            cls.providers = {}
        else:
            cls.register_provider(cls)

        return cls


class ClassProvider(BaseProvider, metaclass=ClassProviderMeta):
    _is_base_provider = True

    @classmethod
    def get_provider_name(cls):
        return getattr(cls, '__provider__', cls.__name__)

    @classmethod
    def register_provider(cls, provider):
        provider_name = cls.get_provider_name()
        if not provider_name:
            return
        cls.providers[provider_name] = provider


def provide(service):
    return ProvidedWrapper(service)
