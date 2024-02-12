# Copyright (c) 2019-2024 Intel Corporation
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

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

from omz_tools import _common

def copy_data(output_dir):
    data_path = output_dir / 'data'

    print('Copying files from {} to {}'.format(_common.PACKAGE_DIR / 'data', data_path))

    shutil.copytree(
        str(_common.PACKAGE_DIR / 'data'),
        str(data_path),
    )


def read_dataset_classes(file_name: Path) -> Dict[int, str]:
    file_path = _common.PACKAGE_DIR / 'data' / 'dataset_classes' / file_name

    if not file_path.exists():
        raise FileNotFoundError(f'File with name: {str(file_name)} not found')

    if file_path.suffix == '.txt':
        classes = _read_txt_classes(file_path)
    elif file_path.suffix == '.json':
        classes = _read_json_classes(file_path)
    else:
        raise NotImplementedError(f'Not supported file extension: {file_path.suffix}')

    return dict(enumerate(classes))


def _read_txt_classes(file_path: Path) -> List[str]:
    with file_path.open() as file:
        return file.read().splitlines()


def _read_json_classes(file_path: Path) -> List[str]:
    with file_path.open() as file:
        # expected json with an array of strings
        return json.load(file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=Path, metavar='DIR',
        default=Path.cwd(), help='Path where to save dataset files')
    args = parser.parse_args()

    try:
        copy_data(args.output_dir)
    except Exception as exp:
        sys.exit('Errors occurred: ' + str(exp))

if __name__ == '__main__':
    main()
