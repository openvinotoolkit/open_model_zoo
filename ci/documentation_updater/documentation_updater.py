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

LOG_LEVELS = [
    'CRITICAL',
    'ERROR',
    'WARNING',
    'INFO',
    'DEBUG',
]


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--model-dir', type=Path, default='../../models',
                        help='Path to root directory with models documentation and configuration files')
    parser.add_argument('--mode', type=str, choices=MODES, default='check',
                        help='Script work mode: "check" only finds diffs, "update" - updates values')
    parser.add_argument('--log-level', choices=LOG_LEVELS, default='WARNING',
                        help='Level of logging')
    parser.add_argument('--ignored-files', type=str,
                        help='List of files which will be ignored')
    parser.add_argument('--ignored-files-list', type=Path,
                        help='Path to file with ignored files')
    args = parser.parse_args()

    if not args.model_dir.is_dir():
        logging.critical("Directory {} does not exist. Please check '--model-dir' option."
                         .format(args.model_dir))
        exit(1)
    if args.ignored_files_list and not args.ignored_files_list.is_file():
        logging.critical("File {} does not exist. Please check '--ignored-files-list' option."
                         .format(args.ignored_files_list))
        exit(1)
    return args


def collect_readme(directory, ignored_files=('index.md',)):
    files = {file.stem: file for file in directory.glob('**/*.md') if file.name not in ignored_files}
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
        with file.open("r", encoding="utf-8") as readme:
            for line in readme:
                if line.startswith('##'):
                    if not started:
                        started = True
                        continue
                    else:
                        break
                if started:
                    desc.append(line)
        desc = convert(desc)
        if desc != '':
            descriptions[name] = desc
        else:
            logging.warning('No description found in {} file. '
                            'Check  compliance with the OMZ Contribution Guide.'.format(file))
    return descriptions


def get_models_from_configs(directory):
    yaml = ruamel.yaml.YAML()
    yaml.preserve_quotes = True
    models = {}
    model_configs = directory.glob('**/model.yml')
    for model in model_configs:
        with model.open("r") as file:
            models[model.parent.name] = (model, yaml.load(file))

    return models


def update_model_descriptions(models, descriptions, mode):
    update_models = []
    for name, desc in descriptions.items():
        model = models.get(name, None)
        if model is None:
            logging.warning('For description file {}.md no model found'.format(name))
            continue
        model = model[1]
        if model['description'] != desc:
            if mode == 'update':
                model['description'] = FoldedScalarString(desc)
            else:
                logging.warning('Found diff for {} model'.format(name))
                logging.debug('\n{:12s}{}\n\tvs\n{:12s}{}'
                              .format('In config:', model['description'], 'In readme:', desc))
            update_models.append(name)
    if mode == 'update':
        logging.info('Description updated for {} models'.format(len(update_models)))
    else:
        logging.info('Description differs for {} models'.format(len(update_models)))
    return update_models


def update_model_configs(models, descriptions, mode):
    diffs = update_model_descriptions(models, descriptions, mode)
    if mode == 'update':
        for name, model in models.items():
            if name in diffs:
                yaml = ruamel.yaml.YAML()
                yaml.indent(mapping=2, sequence=4, offset=2)
                yaml.width = 80
                with open(model[0], "w") as file:
                    yaml.dump(model[1], file)
    return len(diffs)


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
    models = get_models_from_configs(args.model_dir)
    diffs = update_model_configs(models, descriptions, args.mode)

    return diffs


if __name__ == '__main__':
    main()
