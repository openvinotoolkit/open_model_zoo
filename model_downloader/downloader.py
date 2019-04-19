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
import hashlib
import re
import requests
import shlex
import shutil
import ssl
import sys
import tempfile
import time

from pathlib import Path

import common

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

def try_retrieve_from_cache(cache, files):
    try:
        if all(cache.has(file[0]) for file in files):
            for hash, destination in files:
                print('========= Retrieving {} from the cache'.format(destination))
                cache.get(hash, destination)
            print()
            return True
    except Exception as e:
        print(e)
        print('########## Warning: Cache retrieval failed; falling back to downloading ##########')
        print()

    return False

def try_update_cache(cache, hash, source):
    try:
        cache.put(hash, source)
    except Exception as e:
        print(e)
        print('########## Warning: Failed to update the cache ##########')

def try_retrieve(name, destination, expected_hash, cache, num_attempts, start_download):
    if try_retrieve_from_cache(cache, [[expected_hash, destination]]):
        return

    print('========= Downloading {}'.format(destination))

    with destination.open('w+b') as f:
        if try_download(name, f, num_attempts, start_download):
            f.seek(0)
            if verify_hash(f, expected_hash, destination, name):
                try_update_cache(cache, expected_hash, destination)

    print('')

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def postprocess_regex_replace(postproc, top_name, output_dir):
    postproc_file = output_dir / postproc.file

    print('========= Replacing text in {} ========='.format(postproc_file))

    postproc_file_text = postproc_file.read_text()

    orig_file = postproc_file.with_name(postproc_file.name + '.orig')
    if not orig_file.exists():
        postproc_file.replace(orig_file)

    postproc_file_text, num_replacements = postproc.pattern.subn(
        postproc.replacement, postproc_file_text, count=postproc.count)

    if num_replacements == 0:
        sys.exit('Invalid pattern: no occurrences found')

    if postproc.count != 0 and num_replacements != postproc.count:
        sys.exit('Invalid pattern: expected at least {} occurrences, but only {} found'.format(
            postproc.count, num_replacements))

    postproc_file.write_text(postproc_file_text)

def postprocess_unpack_archive(postproc, top_name, output_dir):
    postproc_file = output_dir / postproc.file

    print('========= Unpacking {} ========='.format(postproc_file))

    shutil.unpack_archive(str(postproc_file), str(output_dir), postproc.format)

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
cache = NullCache() if args.cache_dir is None else DirCache(args.cache_dir)
topologies = common.load_topologies_from_args(parser, args)

print('')
print('###############|| Downloading topologies ||###############')
print('')
with requests.Session() as session:
    for top in topologies:
        output = args.output_dir / top.subdir
        output.mkdir(parents=True, exist_ok=True)

        for top_file in top.files:
            destination = output / top_file.name

            if isinstance(top_file.source, common.FileSourceHttp):
                start_download = lambda: start_download_generic(
                    session, top_file.source.url, top_file.size)
            elif isinstance(top_file.source, common.FileSourceGoogleDrive):
                start_download = lambda: start_download_google_drive(
                    session, top_file.source.id, top_file.size)
            else:
                assert False, "Unreachable code"

            try_retrieve(top.name, destination, top_file.sha256, cache, args.num_attempts, start_download)

            if top.name in failed_topologies:
                shutil.rmtree(str(output))
                break

print('')
print('###############|| Post processing ||###############')
print('')
for top in topologies:
    output = args.output_dir / top.subdir

    for postproc in top.postprocessing:
        if isinstance(postproc, common.PostprocRegexReplace):
            postprocess_regex_replace(postproc, top.name, output)
        elif isinstance(postproc, common.PostprocUnpackArchive):
            postprocess_unpack_archive(postproc, top.name, output)
        else:
            assert False, "Unreachable code"

if failed_topologies:
    print('FAILED:')
    print(*sorted(failed_topologies), sep='\n')
    sys.exit(1)
