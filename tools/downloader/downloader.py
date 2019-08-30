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

CHUNK_SIZE = 1 << 15 if sys.stdout.isatty() else 1 << 20

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
                    percent = str(progress_size * 100 // size)

                    print('... %s%%, %d KB, %d KB/s, %d seconds passed' %
                            (percent, progress_size / 1024, speed, duration),
                        end='\r' if sys.stdout.isatty() else '\n', flush=True)

                file.write(chunk)
    finally:
        if sys.stdout.isatty():
            print()

def try_download(file, num_attempts, start_download, size):
    for attempt in range(num_attempts):
        if attempt != 0:
            retry_delay = 10
            print("Will retry in {} seconds...".format(retry_delay))
            time.sleep(retry_delay)

        try:
            chunk_iterable = start_download()
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

    return False

def verify_hash(file, expected_hash, path, model_name):
    actual_hash = hashlib.sha256()
    while True:
        chunk = file.read(1 << 20)
        if not chunk: break
        actual_hash.update(chunk)

    if actual_hash.digest() != bytes.fromhex(expected_hash):
        print('########## Error: Hash mismatch for "{}" ##########'.format(path))
        print('##########     Expected: {}'.format(expected_hash))
        print('##########     Actual:   {}'.format(actual_hash.hexdigest()))
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
                print('========= Retrieving {} from the cache'.format(destination), flush=True)
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

def try_retrieve(name, destination, model_file, cache, num_attempts, start_download):
    destination.parent.mkdir(parents=True, exist_ok=True)

    if try_retrieve_from_cache(cache, [[model_file.sha256, destination]]):
        return True

    print('========= Downloading {}'.format(destination))

    success = False

    with destination.open('w+b') as f:
        if try_download(f, num_attempts, start_download, model_file.size):
            f.seek(0)
            if verify_hash(f, model_file.sha256, destination, name):
                try_update_cache(cache, model_file.sha256, destination)
                success = True

    print('')
    return success

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

def main():
    parser = DownloaderArgumentParser()
    parser.add_argument('-c', '--config', type=Path, metavar='CONFIG.YML',
        help='model configuration file (deprecated)')
    parser.add_argument('--name', metavar='PAT[,PAT...]',
        help='download only models whose names match at least one of the specified patterns')
    parser.add_argument('--list', type=Path, metavar='FILE.LST',
        help='download only models whose names match at least one of the patterns in the specified file')
    parser.add_argument('--all',  action='store_true', help='download all available models')
    parser.add_argument('--print_all', action='store_true', help='print all available models')
    parser.add_argument('-o', '--output_dir', type=Path, metavar='DIR',
        default=Path.cwd(), help='path where to save models')
    parser.add_argument('--cache_dir', type=Path, metavar='DIR',
        help='directory to use as a cache for downloaded files')
    parser.add_argument('--num_attempts', type=positive_int_arg, metavar='N', default=1,
        help='attempt each download up to N times')

    args = parser.parse_args()
    cache = NullCache() if args.cache_dir is None else DirCache(args.cache_dir)
    models = common.load_models_from_args(parser, args)

    failed_models = set()

    print('')
    print('###############|| Downloading models ||###############')
    print('')
    with requests.Session() as session:
        for model in models:
            output = args.output_dir / model.subdirectory
            output.mkdir(parents=True, exist_ok=True)

            for model_file in model.files:
                destination = output / model_file.name

                if not try_retrieve(model.name, destination, model_file, cache, args.num_attempts,
                        lambda: model_file.source.start_download(session, CHUNK_SIZE)):
                    shutil.rmtree(str(output))
                    failed_models.add(model.name)
                    break

    print('')
    print('###############|| Post processing ||###############')
    print('')
    for model in models:
        if model.name in failed_models: continue

        output = args.output_dir / model.subdirectory

        for postproc in model.postprocessing:
            postproc.apply(output)

    if failed_models:
        print('FAILED:')
        print(*sorted(failed_models), sep='\n')
        sys.exit(1)

if __name__ == '__main__':
    main()
