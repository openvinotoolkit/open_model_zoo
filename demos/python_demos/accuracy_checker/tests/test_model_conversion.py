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
import sys
from pathlib import Path

from accuracy_checker.launcher.model_conversion import (exec_mo_binary, find_dlsdk_ir, find_mo, prepare_args)
from tests.common import mock_filesystem


def test_mock_file_system():
    hierarchy = [
        'foo/bar',
        'foo/baz/'
    ]
    with mock_filesystem(hierarchy) as prefix:
        assert (Path(prefix) / 'foo' / 'bar').is_file()
        assert (Path(prefix) / 'foo' / 'baz').is_dir()


def test_find_mo():
    mo_hierarchy = ['deployment_tools/model_optimizer/mo.py']

    with mock_filesystem(mo_hierarchy) as prefix:
        path = Path(prefix)

        assert find_mo([(path / 'deployment_tools' / 'model_optimizer').as_posix()])


def test_find_mo_is_none_when_not_exist():
    mo_hierarchy = ['deployment_tools/model_optimizer/mo.py']

    with mock_filesystem(mo_hierarchy) as prefix:
        path = Path(prefix)

        assert find_mo([(path / 'deployment_tools').as_posix()]) is None


def test_find_mo_list_not_corrupted():
    mo_hierarchy = ['deployment_tools/model_optimizer/mo.py']

    with mock_filesystem(mo_hierarchy) as prefix:
        search_paths = [prefix]

        find_mo(search_paths)

        assert len(search_paths) == 1


def test_find_ir__in_root():
    with mock_filesystem(['model.xml', 'model.bin']) as root:
        model, weights = find_dlsdk_ir(root)
        assert model.endswith('.xml')
        assert weights.endswith('.bin')


def test_find_ir__in_subdir():
    with mock_filesystem(['foo/model.xml', 'foo/model.bin']) as root:
        model, weights = find_dlsdk_ir(root)
        assert model.endswith('.xml')
        assert weights.endswith('.bin')


def test_find_ir__not_found():
    with mock_filesystem(['foo/']) as root:
        model, weights = find_dlsdk_ir(root)
        assert model is None
        assert weights is None


def test_prepare_args():
    args = prepare_args('foo', ['-a', '-b'], {'--bar': 123, '-x': 'baz'})
    assert args[0] == sys.executable
    assert args[1] == 'foo'
    assert '-a' in args
    assert '-b' in args
    assert '--bar' in args
    assert '-x' in args

    assert args[args.index('--bar') + 1] == '123'
    assert args[args.index('-x') + 1] == 'baz'


def test_exec_mo_binary(mocker):
    subprocess_run = mocker.patch('subprocess.run')
    mocker.patch('os.chdir')

    args = prepare_args('ModelOptimizer', value_options={'--foo': 'bar'})
    exec_mo_binary(args)

    subprocess_run.assert_called_once_with(args, check=False, timeout=None)
