import argparse
import logging
import os
import re
import ruamel.yaml
from ruamel.yaml.scalarstring import FoldedScalarString
from sys import exit

REGIMES = [
    'check',
    'update'
    ]

REGIME = ''

LOG_LEVELS = {
    'critical': logging.CRITICAL,
    'error': logging.ERROR,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
}


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--readme-dir', type=str, default='models',
                        help='Path to root directory with models descriptions')
    parser.add_argument('-c', '--config-dir', type=str,
                        help='Path to root directory with topologies configs '
                             '(by default used directory from "--readme-dir" key')

    parser.add_argument('--deprecated_representation', type=bool, default=False,
                        help="Used for old topology's representation")
    parser.add_argument('-l', '--list', type=str, default='list_topologies.yml',
                        help='DEPRECATED: file with topologies list')
    parser.add_argument('-o', '--out-file', type=str,
                        help='DEPRECATED: output file with topologies list '
                             '(by default used original file from --out-file key)')

    parser.add_argument('--regime', type=str, choices=REGIMES, default='check',
                        help='Script work regime: "check" only finds diffs, "update" - updates values')
    parser.add_argument('--log-level', choices=LOG_LEVELS.keys(), default='warning',
                        help='Level of logging')
    return parser.parse_args()


def collect_readme(directory, ignored_files=('index.md',)):
    if not os.path.isdir(directory):
        logging.critical("Directory {} does not exist. Please check '--readme-dir' option.".format(os.path.abspath(directory)))
        exit(1)
    files = {}
    for r, d, f in os.walk(directory):
        for file in f:
            if '.md' not in file or file in ignored_files:
                continue
            files[file.replace(".md", "")] = os.path.join(r, file)
    logging.info('Collected {} readme files'.format(len(files)))
    if not files:
        logging.error("No markdown file found in {}. Exceptions - {}. Ensure, that you set right directory.".format(directory, ignored_files))
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
        readme = open(file, "r")
        for line in readme:
            if line[0:2] == '##':
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
    if not os.path.isdir(directory):
        logging.critical("Directory {} does not exist. Please check '--config-dir' option.".format(os.path.abspath(directory)))
        exit(1)
    yaml = ruamel.yaml.YAML()
    yaml.preserve_quotes = True
    topologies = {}
    for r, d, f in os.walk(directory):
        for file in f:
            if file == 'model.yml':
                topologies[os.path.join(r, file)] = yaml.load(open(os.path.join(r, file), "r"))
                topologies[os.path.join(r, file)]['name'] = r.split('/')[-1]
    return topologies


def get_topologies(file):
    yaml = ruamel.yaml.YAML()
    yaml.preserve_quotes = True
    try:
        topologies = yaml.load(open(file, 'r'))
        return topologies
    except FileNotFoundError as e:
        logging.error('File with list of topologies {} not found'.format(file))
        quit(-1)


def find_models(topologies, name):
    models = []
    for model in topologies:
        if name == model['name']:
            models.append(model)
    return models
    

def update_topologies(topologies, descriptions, compare=lambda lhs, rhs: rhs == lhs):
    updated_models_count = 0
    updated_models = []
    for name, desc in descriptions.items():
        models = find_models(topologies, name)
        if models is None:
            logging.warning('For description file {}.md no model found in topologies list'.format(name))
            continue
        for model in models:
            if not compare(model['description'], desc):
                if REGIME == 'update':
                    model['description'] = FoldedScalarString(desc)
                    updated_models.append(model['name'])
                else:
                    logging.warning('Found diff for {} model'.format(model['name']))
                    logging.debug('\n{:12s}{}\n\tvs\n{:12s}{}'
                                  .format('In config:', model['description'], 'In readme:', desc))
                updated_models_count += 1
    if REGIME == 'update':
        logging.info('Description updated for {} models'.format(updated_models_count))
    else:
        logging.info('Description differs for {} models'.format(updated_models_count))
    return updated_models, updated_models_count


def update_topologies_list(topologies, description):
    tops = topologies['topologies']
    _, diff_count = update_topologies(tops, description)
    return diff_count


def update_topologies_configs(topologies, description):
    diffs, diff_count = update_topologies(topologies.values(), description)
    for filename, topology in topologies.items():
        if topology['name'] in diffs:
            top = topology.copy()
            del top['name']
            save_topologies(filename, top)
    return diff_count


def save_topologies(file, topologies):
    yaml = ruamel.yaml.YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 80
    yaml.dump(topologies, open(file, 'w'))


def main():
    args = parse()
    logging.basicConfig(level=LOG_LEVELS[args.log_level], format='%(levelname)s: %(message)s')
    global REGIME
    REGIME = args.regime

    descriptions = collect_descriptions(collect_readme(args.readme_dir))
    if not args.deprecated_representation:
        config_dir = args.config_dir if args.config_dir else args.readme_dir
        topologies = get_topologies_from_configs(config_dir)
        diffs = update_topologies_configs(topologies, descriptions)
    else:
        topologies = get_topologies(args.list)
        diffs = update_topologies_list(topologies, descriptions)
        save_topologies(args.out_file, topologies)

    return diffs


if __name__ == '__main__':
    main()
