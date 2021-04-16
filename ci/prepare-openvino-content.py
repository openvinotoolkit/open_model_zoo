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

"""
This script prepares OMZ content for OpenVINO toolkit packages. For a given
package type, it will create subdirectories for each OMZ component of the
toolkit in the output directory and place files that belong in that package
and that component into that subdirectory.

The script determines which files belong to each package and component by
using Git attributes that begin with `omz.package`.
"""

import argparse
import subprocess

from pathlib import Path, PurePath

OMZ_ROOT = Path(__file__).resolve().parents[1]

PACKAGE_ATTR_NAME = 'omz.package'
PACKAGE_COMPONENT_ATTR_NAME = 'omz.package.component'

def group_by_n(iterable, n):
    return zip(*[iter(iterable)] * n)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('package', choices=('l', 'm', 'w'), help='which package to build')
    parser.add_argument('output_dir', type=Path, help='where to put package contents')
    args = parser.parse_args()

    ls_files_output = subprocess.check_output(
        ['git', '-C', str(OMZ_ROOT), 'ls-files', '-z'])

    all_files = ls_files_output.decode()[:-1].split('\0')

    check_attr_output = subprocess.check_output(
        ['git', '-C', str(OMZ_ROOT), 'check-attr', '--stdin', '-z',
            PACKAGE_ATTR_NAME, PACKAGE_COMPONENT_ATTR_NAME],
        input=ls_files_output)

    all_attributes = {
        (path, attribute): value
        for path, attribute, value in
            group_by_n(check_attr_output.decode()[:-1].split('\0'), 3)
    }

    def file_is_in_current_package(path):
        package_attr_value = all_attributes[(path, PACKAGE_ATTR_NAME)]

        if package_attr_value == 'unset': # the file is not in any package
            return False

        if package_attr_value in {'set', 'unspecified'}: # the file is in every package
            return True

        return args.package in package_attr_value.split(',')

    files_per_component = {}

    for path in all_files:
        if not file_is_in_current_package(path): continue

        component = all_attributes[(path, PACKAGE_COMPONENT_ATTR_NAME)]
        if component in {'set', 'unspecified', 'unset'}:
            raise RuntimeError(
                f'{path}: {PACKAGE_COMPONENT_ATTR_NAME} attribute must not'
                    ' be unset, unspecified or set with no value')

        files_per_component.setdefault(component, []).append(path)

    if args.package == 'w':
        eol = 'crlf'
        eol_chars = '\r\n'
    else:
        eol = 'lf'
        eol_chars = '\n'

    # copy appropriate files to each component

    for component_name, component_files in files_per_component.items():
        component_output_dir = args.output_dir / component_name / 'deployment_tools/open_model_zoo'
        component_output_dir.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            ['git', '-C', str(OMZ_ROOT),
                '-c', 'core.autocrlf=false', '-c', f'core.eol={eol}', '-c', 'core.symlinks=true',
                'checkout-index', '--stdin', '-z', f'--prefix={component_output_dir.resolve()}/'],
            input=''.join(path + '\0' for path in component_files).encode(),
            check=True)

    # create version.txt

    rev_parse_output = subprocess.check_output(
        ['git', '-C', str(OMZ_ROOT), 'rev-parse', 'HEAD'])
    omz_commit = rev_parse_output.decode().rstrip('\n')

    version_txt_path = args.output_dir / 'tools/deployment_tools/open_model_zoo/version.txt'

    with open(version_txt_path, 'w', newline=eol_chars) as version_txt_file:
        print(omz_commit, file=version_txt_file)

    # create compatibility symlinks

    compat_symlinks = [
        ('deployment_tools/intel_models', 'open_model_zoo/models/intel'),
        ('deployment_tools/open_model_zoo/intel_models', 'models/intel'),
        ('deployment_tools/tools/model_downloader', '../open_model_zoo/tools/downloader'),
        ('deployment_tools/inference_engine/demos', '../open_model_zoo/demos'),
    ]

    for link_source, link_target in compat_symlinks:
        link_source_path = args.output_dir / 'compat_symlinks' / link_source
        link_source_path.parent.mkdir(parents=True, exist_ok=True)
        link_source_path.symlink_to(PurePath(link_target), target_is_directory=True)

if __name__ == '__main__':
    main()
