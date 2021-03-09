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
    parser.add_argument('-d', '--model-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'models',
                        help='Path to root directory with models documentation and configuration files')
    parser.add_argument('--mode', type=str, choices=MODES, default='check',
                        help='Script work mode: "check" only finds diffs, "update" - updates values')
    parser.add_argument('--log-level', choices=LOG_LEVELS, default='WARNING',
                        help='Level of logging')
    args = parser.parse_args()

    if not args.model_dir.is_dir():
        logging.critical("Directory {} does not exist. Please check '--model-dir' option."
                         .format(args.model_dir))
        exit(1)
    return args


def collect_readme(directory, ignored_files):
    files = {file.stem: file for file in directory.glob('**/*.md') if file.name not in ignored_files}
    logging.info('Collected {} description files'.format(len(files)))
    if not files:
        logging.error("No markdown file found in {}. Exceptions - {}. Ensure, that you set right directory."
                      .format(directory, ignored_files))
        exit(1)
    return files


def convert(lines):
    result = ''
    list_signs = ['-', '*']
    for line in lines:
        if len(line.strip()) == 0:
            result += '\n'
        elif line.lstrip()[0] in list_signs:
            result += '\n'
        if len(result) > 0 and not result.endswith('\n'):
            result += ' '
        result += line.rstrip('\n')
    result = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1 <\2>", result) # Links transformation
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
        with model.open("r", encoding="utf-8") as file:
            models[model.parent.name] = (model, yaml.load(file))
        if not models[model.parent.name][1]:
            logging.error("File {} is empty. It will be ignored.".format(model))
            del models[model.parent.name]

    return models


def update_model_descriptions(models, descriptions, mode):
    update_models = []
    missed_models = []
    for name, desc in descriptions.items():
        model = models.get(name, None)
        if model is None:
            logging.error('For description file {}.md no model found'.format(name))
            missed_models.append(name)
            continue
        if not model[1].get('description', None):
            logging.error('No description found in {} for {} model'.format(model[0], name))
            missed_models.append(name)
            continue

        model = model[1]
        if model.get('description', '') != desc:
            if mode == 'update':
                model['description'] = FoldedScalarString(desc)
            else:
                logging.debug('Found diff for {} model'.format(name))
                logging.debug('\n{:12s}{}\n\tvs\n{:12s}{}'
                              .format('In config:', model['description'], 'In readme:', desc))
            update_models.append(name)
    if mode == 'update':
        msg = 'Description updated for {} models, missed for {} models.'
        msg_model_list = 'UPDATED:\n\t{}'
    else:
        msg = 'Description differs for {} models, missed for {} models.'
        msg_model_list = 'DIFFERENCE:\n\t{}'
    logging.info(msg.format(len(update_models), len(missed_models)))
    if len(update_models) > 0:
        logging.info(msg_model_list.format("\n\t".join(update_models)))
    if len(missed_models) > 0:
        logging.info('FAILED:\n\t{}'.format("\n\t".join(missed_models)))
    return update_models


def update_model_configs(models, descriptions, mode):
    diffs = update_model_descriptions(models, descriptions, mode)
    if mode == 'update':
        for name in diffs:
            model = models[name]
            yaml = ruamel.yaml.YAML()
            yaml.indent(mapping=2, sequence=4, offset=2)
            yaml.width = 80
            with model[0].open("w", encoding="utf-8") as file:
                yaml.dump(model[1], file)


def main():
    args = parse()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format='%(levelname)s: %(message)s')

    ignored_files = ('index.md',)
    descriptions = collect_descriptions(collect_readme(args.model_dir, ignored_files))
    models = get_models_from_configs(args.model_dir)
    update_model_configs(models, descriptions, args.mode)


if __name__ == '__main__':
    main()
