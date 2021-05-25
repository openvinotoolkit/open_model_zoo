#!/usr/bin/env python3

# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
This script prepares the OMZ documentation for inclusion into the OpenVINO
documentation. This currently involves two things:

* Inserting Doxygen page IDs into all pages. Without them, Doxygen autogenerates
  ID based on the Markdown file paths. The advantages of explicit IDs are that:

  a) the Doxygen page URLs look nicer;
  b) if we decide to change the layout of the documentation files, we can
     do so without affecting the page IDs (by updating this script so that it
     assigns the same IDs as before), thus not breaking any documentation links.

* Generating a Doxygen layout file. Various sections of the layout are assigned
  xml:id attributes so that they can be inserted into the appropriate parent
  sections when the overall OpenVINO toolkit layout file is generated.

The script also runs some basic checks on the documentation contents.

Install dependencies from requirements-documentation.in to use this script.
'''

import argparse
import logging
import re
import shutil
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from pathlib import Path

import yaml

OMZ_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(OMZ_ROOT / 'ci/lib'))

import omzdocs

XML_ID_ATTRIBUTE = '{http://www.w3.org/XML/1998/namespace}id'

# For most task types, taking the machine-readable form and replacing
# underscores with spaces yields a human-readable English description,
# but for a few types it's useful to override it to improve consistency
# and grammar.
# TODO: consider providing these descriptions through the info dumper.
HUMAN_READABLE_TASK_TYPES = {
    'detection': 'Object Detection',
    'object_attributes': 'Object Attribute Estimation',
    'text_to_speech': 'Text-to-speech',
}

def add_page(output_root, parent, *, id=None, path=None, title=None):
    if parent.tag == 'tab':
        parent.attrib['type'] = 'usergroup'

    element = ET.SubElement(parent, 'tab', type='user', title=title, url='@ref ' + id if id else '')

    if not path:
        assert title, "title must be specified if path isn't"

        element.attrib['title'] = title
        return element

    output_path = output_root / path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (OMZ_ROOT / path).open('r', encoding='utf-8') as input_file:
        lines = input_file.readlines()

    page = omzdocs.DocumentationPage(''.join(lines))

    if page.title is None:
        raise RuntimeError(f'{path}: must begin with level 1 heading')

    if not title:
        title = page.title

    element.attrib['title'] = title

    # the only way to override the ID that Doxygen gives Markdown pages
    # is to add a label to the top-level heading. For simplicity, we hardcode
    # the assumption that the file immediately begins with that heading.
    if not lines[0].startswith('# '):
        raise RuntimeError(f'{path}: line 1 must contain the level 1 heading')

    assert id, "id must be specified if path is"
    lines[0] = lines[0].rstrip('\n') + f' {{#{id}}}\n'

    with (output_root / path).open('w', encoding='utf-8') as output_file:
        output_file.writelines(lines)

    # copy all referenced images
    image_urls = [ref.url for ref in page.external_references() if ref.type == 'image']

    for image_url in image_urls:
        parsed_image_url = urllib.parse.urlparse(image_url)
        if parsed_image_url.scheme or parsed_image_url.netloc:
            continue # not a relative URL

        image_rel_path = path.parent / urllib.request.url2pathname(parsed_image_url.path)

        (output_root / image_rel_path.parent).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(OMZ_ROOT / image_rel_path, output_root / image_rel_path)

    return element


def sort_titles(element):
    element[:] = sorted(element, key=lambda child: child.attrib['title'])


def add_accuracy_checker_pages(output_root, parent_element):
    ac_group_element = add_page(output_root, parent_element,
        id='omz_tools_accuracy_checker', path='tools/accuracy_checker/README.md',
        title='Accuracy Checker Tool')
    ac_group_element.attrib[XML_ID_ATTRIBUTE] = 'omz_tools_accuracy_checker'

    for md_path in OMZ_ROOT.glob('tools/accuracy_checker/*/**/*.md'):
        md_path_rel = md_path.relative_to(OMZ_ROOT)

        if md_path_rel.stem == 'README':
            id_suffix = md_path_rel.parent.name
        elif md_path_rel.stem.endswith('_readme'):
            id_suffix = md_path_rel.stem[:-7]
        else:
            raise RuntimeError('{}: unexpected documentation file name')

        add_page(output_root, ac_group_element,
            id=f'omz_tools_accuracy_checker_{id_suffix}', path=md_path_rel)

    sort_titles(ac_group_element)


def add_model_pages(output_root, parent_element, group, group_title):
    group_element = add_page(output_root, parent_element, title=group_title,
        id=f'omz_models_group_{group}', path=f'models/{group}/index.md')

    task_type_elements = {}

    for md_path in sorted(OMZ_ROOT.glob(f'models/{group}/*/**/*.md')):
        md_path_rel = md_path.relative_to(OMZ_ROOT)

        model_name = md_path_rel.parts[2]

        expected_md_path = Path('models', group, model_name, 'README.md')

        if md_path_rel != expected_md_path:
            raise RuntimeError(f'{md_path_rel}: unexpected documentation file,'
                ' should be {expected_md_path}')

        # FIXME: use the info dumper to query model information instead of
        # parsing the configs. We're not doing that now, because the info
        # dumper doesn't support composite models yet.
        model_yml_path = OMZ_ROOT / 'models' / group / model_name / 'model.yml'
        composite_model_yml_path = model_yml_path.with_name('composite-model.yml')

        if model_yml_path.exists():
            expected_title = model_name

            with open(model_yml_path, 'rb') as f:
                config = yaml.safe_load(f)

            task_type = config['task_type']
        elif composite_model_yml_path.exists():
            expected_title = f'{model_name} (composite)'

            with open(composite_model_yml_path, 'rb') as f:
                config = yaml.safe_load(f)

            task_type = config['task_type']
        else:
            logging.warning(
                '{}: no corresponding model.yml or composite-model.yml found; skipping'
                    .format(md_path_rel))
            continue

        if task_type not in task_type_elements:
            human_readable_task_type = HUMAN_READABLE_TASK_TYPES.get(task_type,
                task_type.replace('_', ' ').title())

            task_type_elements[task_type] = add_page(output_root, group_element,
                title=f'{human_readable_task_type} Models')

        # All model names are unique, so we don't need to include the group
        # in the page ID. However, we do prefix "model_", so that model pages
        # don't conflict with any other pages in the omz_models namespace that
        # might be added later.
        page_id = 'omz_models_model_' + re.sub(r'[^a-zA-Z0-9]', '_', model_name)

        model_element = add_page(output_root, task_type_elements[task_type],
            id=page_id, path=md_path_rel)

        if model_element.attrib['title'] != expected_title:
            raise RuntimeError(f'{md_path_rel}: should have title "{expected_title}"')

    sort_titles(group_element)


def main():
    logging.basicConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=Path,
        help='directory to output prepared documentation files into')

    args = parser.parse_args()

    output_root = args.output_dir

    output_root.mkdir(parents=True, exist_ok=True)

    doxygenlayout_element = ET.Element('doxygenlayout', version='1.0')
    navindex_element = ET.SubElement(doxygenlayout_element, 'navindex')

    add_accuracy_checker_pages(output_root, navindex_element)

    datasets_element = add_page(output_root, navindex_element,
        id='omz_data_datasets', path='data/datasets.md')

    # The xml:id here is omz_data rather than omz_data_datasets, because
    # later we might want to have other pages in the "data" directory. If
    # that happens, we'll create a parent page with ID "omz_data" and move
    # the xml:id to that page, thus integrating the new pages without having
    # to change the upstream OpenVINO documentation building process.
    datasets_element.attrib[XML_ID_ATTRIBUTE] = 'omz_data'

    downloader_element = add_page(output_root, navindex_element,
        id='omz_tools_downloader', path='tools/downloader/README.md', title='Model Downloader')
    downloader_element.attrib[XML_ID_ATTRIBUTE] = 'omz_tools_downloader'

    trained_models_group_element = add_page(output_root, navindex_element,
        title="Trained Models")
    trained_models_group_element.attrib[XML_ID_ATTRIBUTE] = 'omz_models'

    add_model_pages(output_root, trained_models_group_element,
        'intel', "Intel's Pre-trained Models")
    add_model_pages(output_root, trained_models_group_element,
        'public', "Public Pre-trained Models")

    demos_group_element = add_page(output_root, navindex_element,
        title="Demos", id='omz_demos', path='demos/README.md')
    demos_group_element.attrib[XML_ID_ATTRIBUTE] = 'omz_demos'

    for md_path in [
        *OMZ_ROOT.glob('demos/*_demo/*/README.md'),
        *OMZ_ROOT.glob('demos/*_demo_*/*/README.md'),
    ]:
        md_path_rel = md_path.relative_to(OMZ_ROOT)

        # <name>_<implementation>
        demo_id = '_'.join(md_path_rel.parts[1:3])

        demo_element = add_page(output_root, demos_group_element,
            id='omz_demos_' + demo_id, path=md_path_rel)

        if not re.search(r'\bDemo\b', demo_element.attrib['title']):
            raise RuntimeError(f'{md_path_rel}: title must contain "Demo"')

    sort_titles(demos_group_element)

    with (output_root / 'DoxygenLayout.xml').open('wb') as layout_file:
        ET.ElementTree(doxygenlayout_element).write(layout_file)

if __name__ == '__main__':
    main()
