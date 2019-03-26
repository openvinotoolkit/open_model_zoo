#!/usr/bin/env python3

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

import argparse
import collections
import fnmatch
import hashlib
import re
import requests
import shlex
import shutil
import ssl
import sys
import tarfile
import tempfile
import time
import yaml

from pathlib import Path

Framework = collections.namedtuple('Framework', ['model_extension', 'weights_extension'])
MemberRequest = collections.namedtuple('MemberRequest', ['path', 'destination', 'expected_hash'])

FRAMEWORKS = {
    'caffe': Framework('.prototxt', '.caffemodel'),
    'dldt': Framework('.xml', '.bin'),
    'mxnet': Framework('.json', '.params'),
    'tf': Framework('.prototxt', '.frozen.pb'),
}

DOWNLOAD_TIMEOUT = 5 * 60

failed_topologies = set()

def process_download(chunk_iterable, size, file):
    start_time = time.monotonic()
    progress_size = 0

    try:
        for chunk in chunk_iterable:
            if chunk:
                duration = time.monotonic() - start_time
                progress_size += len(chunk)
                if duration != 0:
                    speed = progress_size // (1024 * duration)
                    if size == 0:
                        percent = '---'
                    else:
                        percent = str(min(progress_size * 100 // size, 100))
                    print('\r... %s%%, %d KB, %d KB/s, %d seconds passed' %
                            (percent, progress_size / 1024, speed, duration),
                        end='', flush=True)
                file.write(chunk)
    finally:
        print()

def start_download_generic(session, url, total_size=None):
    response = session.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
    response.raise_for_status()

    if total_size is None:
        size = int(response.headers.get('content-length', 0))
    else:
        size = total_size

    return response.iter_content(chunk_size=8192), size

def start_download_google_drive(session, id, total_size):
    chunk_size = 32768

    URL = 'https://docs.google.com/uc?export=download'
    response = session.get(URL, params={'id' : id}, stream=True, timeout=DOWNLOAD_TIMEOUT)
    response.raise_for_status()

    token = get_confirm_token(response)
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params=params, stream=True, timeout=DOWNLOAD_TIMEOUT)

    return response.iter_content(chunk_size=32768), total_size

def try_download(name, file, num_attempts, start_download):
    for attempt in range(num_attempts):
        if attempt != 0:
            retry_delay = 10
            print("Will retry in {} seconds...".format(retry_delay))
            time.sleep(retry_delay)

        try:
            chunk_iterable, size = start_download()
            file.seek(0)
            file.truncate()
            process_download(chunk_iterable, size, file)
            return True
        except requests.exceptions.ConnectionError as e:
            print("Error Connecting:", e)
        except requests.exceptions.Timeout as e:
            print("Timeout Error:", e)
        except requests.exceptions.TooManyRedirects as e:
            print("Redirects Error: requests exceeds maximum number of redirects", e)
        except (requests.exceptions.RequestException, ssl.SSLError) as e:
            print(e)

    failed_topologies.add(name)
    return False

def verify_hash(file, expected_hash, path, top_name):
    actual_hash = hashlib.sha256()
    while True:
        chunk = file.read(1 << 20)
        if not chunk: break
        actual_hash.update(chunk)

    if actual_hash.digest() != bytes.fromhex(expected_hash):
        print('########## Error: Hash mismatch for "{}" ##########'.format(path))
        print('##########     Expected: {}'.format(expected_hash))
        print('##########     Actual:   {}'.format(actual_hash.hexdigest()))
        failed_topologies.add(top_name)
        return False
    return True

class NullCache:
    def has(self, hash): return False
    def put(self, hash, path): pass

class DirCache:
    _FORMAT = 1 # increment if backwards-incompatible changes to the format are made
    _HASH_LEN = hashlib.sha256().digest_size * 2

    def __init__(self, cache_dir):
        self._cache_dir = cache_dir / str(self._FORMAT)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._staging_dir = self._cache_dir / 'staging'
        self._staging_dir.mkdir(exist_ok=True)

    def _hash_path(self, hash):
        hash = hash.lower()
        assert len(hash) == self._HASH_LEN
        assert re.fullmatch('[0-9a-f]+', hash)
        return self._cache_dir / hash[:2] / hash[2:]

    def has(self, hash):
        return self._hash_path(hash).exists()

    def get(self, hash, path):
        shutil.copyfile(str(self._hash_path(hash)), str(path))

    def put(self, hash, path):
        # A file in the cache must have the hash implied by its name. So when we upload a file,
        # we first copy it to a temporary file and then atomically move it to the desired name.
        # This prevents interrupted runs from corrupting the cache.
        with path.open('rb') as src_file:
            with tempfile.NamedTemporaryFile(dir=str(self._staging_dir), delete=False) as staging_file:
                staging_path = Path(staging_file.name)
                shutil.copyfileobj(src_file, staging_file)

        hash_path = self._hash_path(hash)
        hash_path.parent.mkdir(parents=True, exist_ok=True)
        staging_path.replace(self._hash_path(hash))

def try_download_simple(name, destination, expected_hash, cache, num_attempts, start_download):
    if cache.has(expected_hash):
        print('========= Retrieving {} from the cache'.format(destination))
        cache.get(expected_hash, destination)
        print()
        return

    print('========= Downloading {}'.format(destination))

    with destination.open('w+b') as f:
        if try_download(name, f, num_attempts, start_download):
            f.seek(0)
            if verify_hash(f, expected_hash, destination, name):
                cache.put(expected_hash, destination)

    print('')

def try_download_tar(name, members, cache, num_attempts, start_download):
    if all(cache.has(member.expected_hash) for member in members):
        for member in members:
            print('========= Retrieving {} from the cache'.format(member.destination))
            cache.get(member.expected_hash, member.destination)
        print()
        return

    for member in members:
        print('========= Downloading {}'.format(member.destination))

    with tempfile.TemporaryFile() as f:
        if try_download(name, f, num_attempts, start_download):
            f.seek(0)
            tar = tarfile.open(fileobj=f, mode='r:*')
            for member in members:
                try:
                    member_info = tar.getmember(member.path)
                except KeyError:
                    print('########## Error: Archive missing required member "{}" ##########'.format(member.path))
                    print()
                    failed_topologies.add(name)
                    return

                if not member_info.isfile():
                    print('########## Error: Archive member "{}" is not a regular file ##########'.format(member.path))
                    print()
                    failed_topologies.add(name)
                    return

                with tar.extractfile(member_info) as member_file, member.destination.open('w+b') as destination_file:
                    shutil.copyfileobj(member_file, destination_file)

                    destination_file.seek(0)
                    if verify_hash(destination_file, member.expected_hash, member.destination, name):
                        cache.put(member.expected_hash, member.destination)
                    else:
                        break

    print('')

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def delete_param(model):
    tmpfile = args.output_dir / 'tmp.txt'
    with model.open('r') as input_file, tmpfile.open('w') as output_file:
        data=input_file.read()
        updated_data = re.sub(' +save_output_param \{.*\n.*\n +\}\n', '', data, count=1)
        output_file.write(updated_data)
    tmpfile.replace(model)

def layers_to_layer(model):
    tmpfile = args.output_dir / 'tmp.txt'
    with model.open('r') as input_file, tmpfile.open('w') as output_file:
        data=input_file.read()
        updated_data = data.replace('layers {', 'layer {')
        output_file.write(updated_data)
    tmpfile.replace(model)

def change_dim(model, old_dim, new_dim):
    new = 'dim: ' + str(new_dim)
    old = 'dim: ' + str(old_dim)
    tmpfile = args.output_dir / 'tmp.txt'
    with model.open('r') as input_file, tmpfile.open('w') as output_file:
        data=input_file.read()
        data = data.replace(old, new, 1)
        output_file.write(data)
    tmpfile.replace(model)

class DownloaderArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

def positive_int_arg(value_str):
    try:
        value = int(value_str)
        if value > 0: return value
    except ValueError:
        pass

    raise argparse.ArgumentTypeError('must be a positive integer (got {!r})'.format(value_str))

parser = DownloaderArgumentParser(epilog = 'list_topologies.yml - default configuration file')
parser.add_argument('-c', '--config', type = Path, metavar = 'CONFIG.YML',
    default = Path(__file__).resolve().parent / 'list_topologies.yml', help = 'path to YML configuration file')
parser.add_argument('--name', metavar = 'PAT[,PAT...]',
    help = 'download only topologies whose names match at least one of the specified patterns')
parser.add_argument('--list', type = Path, metavar = 'FILE.LST',
    help = 'download only topologies whose names match at least one of the patterns in the specified file')
parser.add_argument('--all',  action = 'store_true', help = 'download all topologies from the configuration file')
parser.add_argument('--print_all', action = 'store_true', help = 'print all available topologies')
parser.add_argument('-o', '--output_dir', type = Path, metavar = 'DIR',
    default = Path.cwd(), help = 'path where to save topologies')
parser.add_argument('--cache_dir', type = Path, metavar = 'DIR',
    help = 'directory to use as a cache for downloaded files')
parser.add_argument('--num_attempts', type = positive_int_arg, metavar = 'N', default = 1,
    help = 'attempt each download up to N times')

args = parser.parse_args()
path_to_config = args.config
cache = NullCache() if args.cache_dir is None else DirCache(args.cache_dir)

with path_to_config.open() as stream:
    try:
        c_new = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        sys.exit('Cannot parse the YML, please check the configuration file')

if args.print_all:
    for top in c_new['topologies']:
        print(top['name'])
    sys.exit()

if sum([args.all, args.name is not None, args.list is not None]) > 1:
    parser.error('Please choose either "--all", "--name" or "--list"')

if args.all:
    topologies = c_new['topologies']
elif args.name is not None or args.list is not None:
    if args.name is not None:
        patterns = args.name.split(',')
    else:
        patterns = []
        with args.list.open() as list_file:
            for list_line in list_file:
                tokens = shlex.split(list_line, comments=True)
                if not tokens: continue

                patterns.append(tokens[0])
                # For now, ignore any other tokens in the line.
                # We might use them as additional parameters later.

    topologies = []
    for pattern in patterns:
        matching_topologies = [top for top in c_new['topologies']
            if fnmatch.fnmatchcase(top['name'], pattern)]

        if not matching_topologies:
            sys.exit('No matching topologies: "{}"'.format(pattern))

        topologies.extend(matching_topologies)
else:
    print('Please choose either "--all", "--name" or "--list"', file = sys.stderr)
    parser.print_help()
    print('')
    print('========== All available topologies ==========')
    print('')
    for top in c_new['topologies']:
        print(top['name'])
    sys.exit(2)

print('')
print('###############|| Downloading topologies ||###############')
print('')
with requests.Session() as session:
    for top in topologies:
        try:
            framework = FRAMEWORKS[top['framework']]
        except KeyError:
            sys.exit('Unknown framework "{}" for topology "{}"'.format(top['framework'], top['name']))

        output = args.output_dir / top['output']
        output.mkdir(parents=True, exist_ok=True)

        model_destination = output / (top['name'] + framework.model_extension)
        weights_destination = output / (top['name'] + framework.weights_extension)

        if {'model_google_drive_id', 'model_size'} <= top.keys():
            try_download_simple(top['name'], model_destination, top['model_hash'], cache, args.num_attempts,
                lambda: start_download_google_drive(session, top['model_google_drive_id'], top['model_size']))
        elif 'model' in top:
            try_download_simple(top['name'], model_destination, top['model_hash'], cache, args.num_attempts,
                lambda: start_download_generic(session, top['model'], top.get('model_size')))

        if {'weights_google_drive_id', 'weights_size'} <= top.keys():
            try_download_simple(top['name'], weights_destination, top['weights_hash'], cache, args.num_attempts,
                lambda: start_download_google_drive(session, top['weights_google_drive_id'], top['weights_size']))
        elif 'weights' in top:
            try_download_simple(top['name'], weights_destination, top['weights_hash'], cache, args.num_attempts,
                lambda: start_download_generic(session, top['weights'], top.get('weights_size')))

        members = []

        if {'model_path_prefix', 'weights_path_prefix'} <= top.keys():
            members.append(MemberRequest(top['model_path_prefix'], model_destination, top['model_hash']))
            members.append(MemberRequest(top['weights_path_prefix'], weights_destination, top['weights_hash']))
        elif 'model_path_prefix' in top:
            members.append(MemberRequest(top['model_path_prefix'], weights_destination, top['model_hash']))

        if {'tar_google_drive_id', 'tar_size'} <= top.keys():
            try_download_tar(top['name'], members, cache, args.num_attempts,
                lambda: start_download_google_drive(session, top['tar_google_drive_id'], top['tar_size']))
        elif 'tar' in top:
            try_download_tar(top['name'], members, cache, args.num_attempts,
                lambda: start_download_generic(session, top['tar'], top.get('tar_size')))

        if top['name'] in failed_topologies:
            shutil.rmtree(str(output))

print('')
print('###############|| Post processing ||###############')
print('')
for top in topologies:
    framework = FRAMEWORKS[top['framework']]

    model_name = top['name'] + framework.model_extension
    weights_name = top['name'] + framework.weights_extension
    output = args.output_dir / top['output']
    path_to_model = output / model_name
    path_to_weights = output / weights_name
    if 'delete_output_param' in top:
        if path_to_model.exists():
            print('========= Deleting "save_output_param" from %s =========' % (model_name))
            delete_param(path_to_model)
    if {'old_dims', 'new_dims'} <= top.keys():
        if path_to_model.exists():
            print('========= Changing input dimensions in %s =========' % (model_name))
            for j in range(len(top['old_dims'])):
                change_dim(path_to_model, top['old_dims'][j], top['new_dims'][j])
    if 'layers_to_layer' in top:
        if path_to_model.exists():
            print('========= Moving to new Caffe layer presentation %s =========' % (model_name))
            layers_to_layer(path_to_model)

if failed_topologies:
    print('FAILED:')
    print(*sorted(failed_topologies), sep='\n')
    sys.exit(1)
