#!/usr/bin/env python3

"""
 Copyright (c) 2020 Intel Corporation

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

import argparse
import logging
import re
import ruamel.yaml
import shlex

from pathlib import Path
from ruamel.yaml.scalarstring import FoldedScalarString
from sys import exit

MODES = [
    'check',
    'update'
    ]

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--model-dir', type=str, default='../../models',
                        help='Path to root directory with models documentation and configuration files')
    parser.add_argument('--mode', type=str, choices=MODES, default='check',
                        help='Script work mode: "check" only finds diffs, "update" - updates values')
    parser.add_argument('--log-level', choices=logging._levelToName.values(), default='WARNING',
                        help='Level of logging')
    parser.add_argument('--ignored-files', type=str,
                        help='List of files which will be ignored')
    parser.add_argument('--ignored-files-list', type=Path,
                        help='Path to file with ignored files')
    return parser.parse_args()


def collect_readme(directory, ignored_files=('index.md',)):
    if not Path(directory).is_dir():
        logging.critical("Directory {} does not exist. Please check '--model-dir' option."
                         .format(directory))
        exit(1)
    files = {}
    md_files = Path(directory).glob('**/*.md')
    for file in md_files:
        if file.name in ignored_files:
            continue
        files[file.stem] = file
    logging.info('Collected {} readme files'.format(len(files)))
    if not files:
        logging.error("No markdown file found in {}. Exceptions - {}. Ensure, that you set right directory."
                      .format(directory, ignored_files))
        exit(1)
    return files


def convert(lines):
    def flatten_links(s):
        links = re.findall(r"\[.*?\]\(.*?\)", s)
        for link in links:
            plain_link = re.sub('[\[\]]', '', link.replace('](', ' (').replace('(', '<').replace(')', '>'))
            s = s.replace(link, plain_link)
        return s

    result = ''
    list_signs = ['-', '*']
    for line in lines:
        if len(line.strip()) == 0:
            result += '\n'
        elif line.lstrip()[0] in list_signs:
            result += '\n'
        if len(result) > 0 and result[len(result)-1] != '\n':
            result += ' '
        result += line.rstrip('\n')
        result = flatten_links(result)
        result = result.replace("`", "\"").replace("\*", "*")
    return result.strip()


def collect_descriptions(files):
    descriptions = {}
    for name, file in files.items():
        started = False
        desc = []
        with open(file, "r", encoding="utf-8") as readme:
            for line in readme:
                if line.startswith('##'):
                    if not started:
                        started = True
                        continue
                    else:
                        break
                if started:
                    desc.append(line)
        desc = convert(desc).strip('\n')
        if desc != '':
            descriptions[name] = desc
        else:
            logging.warning('No description found in {} file. '
                            'Note, that file must be marked and description must be first chapter'.format(file))
    return descriptions


def get_topologies_from_configs(directory):
    if not Path(directory).is_dir():
        logging.critical("Directory {} does not exist. Please check '--config-dir' option.".
                         format(directory))
        exit(1)
    yaml = ruamel.yaml.YAML()
    yaml.preserve_quotes = True
    topologies = {}
    models = Path(directory).glob('**/model.yml')
    for model in models:
        with open(model, "r") as file:
            topologies[model.parent.name] = (model, yaml.load(file))

    return topologies


def update_topologies(topologies, descriptions, mode, compare=lambda lhs, rhs: rhs == lhs):
    updated_models_count = 0
    updated_models = []
    for name, desc in descriptions.items():
        model = topologies.get(name, None)
        if model is None:
            logging.warning('For description file {}.md no model found in topologies list'.format(name))
            continue
        model = model[1]
        if not compare(model['description'], desc):
            if mode == 'update':
                model['description'] = FoldedScalarString(desc)
                updated_models.append(name)
            else:
                logging.warning('Found diff for {} model'.format(name))
                logging.debug('\n{:12s}{}\n\tvs\n{:12s}{}'
                              .format('In config:', model['description'], 'In readme:', desc))
            updated_models_count += 1
    if mode == 'update':
        logging.info('Description updated for {} models'.format(updated_models_count))
    else:
        logging.info('Description differs for {} models'.format(updated_models_count))
    return updated_models, updated_models_count


def update_topologies_configs(topologies, descriptions, mode):
    diffs, diff_count = update_topologies(topologies, descriptions, mode)
    for name, topology in topologies.items():
        if name in diffs:
            save_topology(topology[0], topology[1])
    return diff_count


def save_topology(file, topology):
    yaml = ruamel.yaml.YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 80
    yaml.dump(topology, open(file, 'w'))


def get_ignored_files(args):
    files_to_ignore = ['index.md']
    if args.ignored_files_list:
        if args.ignored_files_list.is_file():
            with args.ignored_files_list.open() as file:
                for line in file:
                    files_to_ignore.extend(shlex.split(line, comments=True))
        else:
            logging.error('File {} not exist. Please, recheck "--ignored-files-list" option'
                          .format(args.ignored_files_list))
    if args.ignored_files:
        files_to_ignore.extend(args.ignored_files.split(','))
    return files_to_ignore


def main():
    args = parse()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format='%(levelname)s: %(message)s')

    descriptions = collect_descriptions(collect_readme(args.model_dir, get_ignored_files(args)))
    topologies = get_topologies_from_configs(args.model_dir)
    diffs = update_topologies_configs(topologies, descriptions, args.mode)

    return diffs


if __name__ == '__main__':
    main()
