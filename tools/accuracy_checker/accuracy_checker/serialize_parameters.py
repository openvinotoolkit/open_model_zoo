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

# pylint:disable=W0614,W0401

import json
from collections import OrderedDict
from argparse import Action, ArgumentParser

from .adapters import *
from .annotation_converters import *
from .metrics import *
from .preprocessor import *
from .postprocessor import *
from .topology_types import *
from .launcher import *
from .logging import print_info


def inheritors(cls, recursively=True):
    subclasses = set()
    work = [cls]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                if recursively:
                    work.append(child)
    return subclasses


def parents(cls, excluded_classes=None):
    cls_parents = set()
    work = [cls]
    while work:
        child = work.pop()
        for parent in child.__bases__:
            if parent not in cls_parents:
                if not excluded_classes or parent not in excluded_classes:
                    cls_parents.add(parent)
                    work.append(parent)
    return cls_parents


def check_topology_is_supported(supported_topology_types, filter_topology_types):
    if list(set(supported_topology_types) & set(filter_topology_types)):
        return True

    supported_topology_classes = [y for x, y in globals().items()
                                  if (hasattr(y, '__provider__') and y.__provider__ in supported_topology_types)]
    filter_topology_classes = [y for x, y in globals().items()
                               if (hasattr(y, '__provider__') and y.__provider__ in filter_topology_types)]
    for cls in supported_topology_classes:
        for filter_cls in filter_topology_classes:
            filter_cls_parents = parents(filter_cls, [GenericTopology])
            filter_cls_childs = inheritors(filter_cls)
            if cls in filter_cls_childs or cls in filter_cls_parents:
                return True
    return False


def add_section(base_class, topology_types=None, representations=None, filtered_providers=None):
    providers = inheritors(base_class)
    class_parameters = OrderedDict()
    for provider in providers:
        if provider.__provider__:
            provider_parameters = OrderedDict()

            supported_topology_types = []
            if hasattr(provider, 'topology_types'):
                supported_topology_types = [x.__provider__ for x in provider.topology_types]
                if topology_types and not check_topology_is_supported(supported_topology_types, topology_types):
                    continue

            if supported_topology_types:
                provider_parameters["topology_types"] = supported_topology_types

            supported_representations = []
            if hasattr(provider, 'annotation_types'):
                supported_representations.extend({x.__name__ for x in provider.annotation_types})
            if hasattr(provider, 'prediction_types'):
                supported_representations.extend({x.__name__ for x in provider.prediction_types})
            if supported_representations:
                if representations and not list(set(supported_representations) & set(representations)):
                    continue
                provider_parameters["representations"] = supported_representations

            if filtered_providers and provider.__provider__ not in filtered_providers:
                continue

            parameters = provider.parameters()
            parameters_json = OrderedDict()
            for key in parameters:
                parameters_json[key] = parameters[key].parameters()
            provider_parameters["parameters"] = parameters_json

            if provider_parameters:
                class_parameters[provider.__provider__] = provider_parameters

    return class_parameters


all_topology_types = [x.__provider__ for x in inheritors(Topology)]
all_launchers = [x.__provider__ for x in inheritors(Launcher)]

valid_values_map_for_action = {
    "topology_types": all_topology_types,
    "launchers": all_launchers
}


class ListAction(Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(ListAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        values = [x.strip() for x in values.split(',')]
        if self.dest in valid_values_map_for_action.keys() and valid_values_map_for_action[self.dest]:
            for v in values:
                if v not in valid_values_map_for_action[self.dest]:
                    raise Exception("Unknown " + self.dest + ': ' + v +
                                    ".\nSelect from list:\n" + '\n'.join(valid_values_map_for_action[self.dest]))
        setattr(namespace, self.dest, values)


def get_recursively(search_dict, field):
    fields_found = set()

    for key, value in search_dict.items():
        if key == field:
            if isinstance(value, list):
                for item in value:
                    fields_found.add(item)
            else:
                fields_found.add(value)

        elif isinstance(value, dict):
            results = get_recursively(value, field)
            for result in results:
                fields_found.add(result)

        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    more_results = get_recursively(item, field)
                    for another_result in more_results:
                        fields_found.add(another_result)

    return fields_found


def add_topology_types(cls):
    topology_descriptions = {}
    topology_descriptions[cls.__name__] = {'description': cls.description}
    child_topology_types = {}
    for child in inheritors(cls, False):
        child_topology_types.update(add_topology_types(child))

    if child_topology_types:
        topology_descriptions[cls.__name__].update({'topology_types': child_topology_types})

    return topology_descriptions


def fetch(topology_types=None, launchers=None):
    if topology_types:
        topology_types = set(sorted(topology_types))

    if launchers:
        launchers = set(sorted(launchers))

    json_dict = {}
    json_dict[Topology.__provider_type__] = add_section(Topology, filtered_providers=topology_types)

    models_dict = {}
    models_dict[Launcher.__provider_type__] = add_section(Launcher, filtered_providers=launchers)
    json_dict['models'] = models_dict

    dataset_dict = {}
    for base_provider in [Adapter, BaseFormatConverter]:
        dataset_dict[base_provider.__provider_type__] = add_section(base_provider, topology_types=topology_types)

    representations = get_recursively(dataset_dict, 'representations')
    for base_provider in [Metric, Preprocessor, Postprocessor]:
        dataset_dict[base_provider.__provider_type__] = add_section(base_provider, representations=representations)

    json_dict['datasets'] = dataset_dict
    return json_dict


def main():
    parser = ArgumentParser(description='Accuracy Checker Parameter Fetcher', allow_abbrev=False)
    parser.add_argument(
        '-t', '--topology_types',
        help='Topology types: ' + ', '.join(all_topology_types),
        required=False,
        action=ListAction,
        default=all_topology_types
    )
    parser.add_argument(
        '-l', '--launchers',
        help='Launchers: ' + ', '.join(all_launchers),
        required=False,
        action=ListAction,
        default=all_launchers
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file',
        required=False,
        default='serialized_parameters.json'
    )

    args = parser.parse_args()

    print_info("Gather parameters for following topology types: " + ", ".join(args.topology_types))
    json_dict = fetch(args.topology_types, args.launchers)
    json_content = json.dumps(json_dict, sort_keys=True, indent=4)

    print_info('Writing to {}...'.format(args.output))
    with open(args.output, 'w') as f:
        f.write(json_content)


if __name__ == '__main__':
    main()
