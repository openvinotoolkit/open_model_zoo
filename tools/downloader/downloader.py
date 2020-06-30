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
import concurrent.futures
import contextlib
import functools
import hashlib
import re
import requests
import shlex
import shutil
import ssl
import sys
import tempfile
import threading
import time
import types

from pathlib import Path

import common

CHUNK_SIZE = 1 << 15 if sys.stdout.isatty() else 1 << 20

def process_download(reporter, chunk_iterable, size, progress, file):
    start_time = time.monotonic()

    try:
        for chunk in chunk_iterable:
            reporter.job_context.check_interrupted()

            if chunk:
                duration = time.monotonic() - start_time
                progress.size += len(chunk)
                progress.hasher.update(chunk)

                if duration != 0:
                    speed = int(progress.size / (1024 * duration))
                else:
                    speed = '?'

                percent = progress.size * 100 // size

                reporter.print_progress('... {}%, {} KB, {} KB/s, {} seconds passed',
                    percent, progress.size // 1024, speed, int(duration))
                reporter.emit_event('model_file_download_progress', size=progress.size)

                file.write(chunk)

                # don't attempt to finish a file if it's bigger than expected
                if progress.size > size:
                    break
    finally:
        reporter.end_progress()

def try_download(reporter, file, num_attempts, start_download, size):
    progress = types.SimpleNamespace(size=0)

    for attempt in range(num_attempts):
        if attempt != 0:
            retry_delay = 10
            reporter.print("Will retry in {} seconds...", retry_delay, flush=True)
            time.sleep(retry_delay)

        try:
            reporter.job_context.check_interrupted()
            chunk_iterable, continue_offset = start_download(offset=progress.size)

            if continue_offset not in {0, progress.size}:
                # Somehow we neither restarted nor continued from where we left off.
                # Try to restart.
                chunk_iterable, continue_offset = start_download(offset=0)
                if continue_offset != 0:
                    reporter.log_error("Remote server refuses to send whole file, aborting")
                    return None

            if continue_offset == 0:
                file.seek(0)
                file.truncate()
                progress.size = 0
                progress.hasher = hashlib.sha256()

            process_download(reporter, chunk_iterable, size, progress, file)

            if progress.size > size:
                reporter.log_error("Remote file is longer than expected ({} B), download aborted", size)
                # no sense in retrying - if the file is longer, there's no way it'll fix itself
                return None
            elif progress.size < size:
                reporter.log_error("Downloaded file is shorter ({} B) than expected ({} B)",
                    progress.size, size)
                # it's possible that we got disconnected before receiving the full file,
                # so try again
            else:
                return progress.hasher.digest()
        except (requests.exceptions.RequestException, ssl.SSLError):
            reporter.log_error("Download failed", exc_info=True)

    return None

def verify_hash(reporter, actual_hash, expected_hash, path):
    if actual_hash != bytes.fromhex(expected_hash):
        reporter.log_error('Hash mismatch for "{}"', path)
        reporter.log_details('Expected: {}', expected_hash)
        reporter.log_details('Actual:   {}', actual_hash.hex())
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

def try_retrieve_from_cache(reporter, cache, files):
    try:
        if all(cache.has(file[0]) for file in files):
            for hash, destination in files:
                reporter.job_context.check_interrupted()

                reporter.print_section_heading('Retrieving {} from the cache', destination)
                cache.get(hash, destination)
            reporter.print()
            return True
    except Exception:
        reporter.log_warning('Cache retrieval failed; falling back to downloading', exc_info=True)
        reporter.print()

    return False

def try_update_cache(reporter, cache, hash, source):
    try:
        cache.put(hash, source)
    except Exception:
        reporter.log_warning('Failed to update the cache', exc_info=True)

def try_retrieve(reporter, destination, model_file, cache, num_attempts, start_download):
    destination.parent.mkdir(parents=True, exist_ok=True)

    if try_retrieve_from_cache(reporter, cache, [[model_file.sha256, destination]]):
        return True

    reporter.print_section_heading('Downloading {}', destination)

    success = False

    with destination.open('w+b') as f:
        actual_hash = try_download(reporter, f, num_attempts, start_download, model_file.size)

    if actual_hash and verify_hash(reporter, actual_hash, model_file.sha256, destination):
        try_update_cache(reporter, cache, model_file.sha256, destination)
        success = True

    reporter.print()
    return success

def download_model(reporter, args, cache, session_factory, requested_precisions, model):
    session = session_factory()
    reporter.emit_event('model_download_begin', model=model.name, num_files=len(model.files))

    output = args.output_dir / model.subdirectory
    output.mkdir(parents=True, exist_ok=True)

    for model_file in model.files:
        if len(model_file.name.parts) == 2:
            p = model_file.name.parts[0]
            if p in common.KNOWN_PRECISIONS and p not in requested_precisions:
                continue

        model_file_reporter = reporter.with_event_context(model=model.name, model_file=model_file.name.as_posix())
        model_file_reporter.emit_event('model_file_download_begin', size=model_file.size)

        destination = output / model_file.name

        if not try_retrieve(model_file_reporter, destination, model_file, cache, args.num_attempts,
                functools.partial(model_file.source.start_download, session, CHUNK_SIZE)):
            shutil.rmtree(str(output))
            model_file_reporter.emit_event('model_file_download_end', successful=False)
            reporter.emit_event('model_download_end', model=model.name, successful=False)
            return False

        model_file_reporter.emit_event('model_file_download_end', successful=True)

    reporter.emit_event('model_download_end', model=model.name, successful=True)
    return True


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


# There is no evidence that the requests.Session class is thread-safe,
# so for safety, we use one Session per thread. This class ensures that
# each thread gets its own Session.
class ThreadSessionFactory:
    def __init__(self, exit_stack):
        self._lock = threading.Lock()
        self._thread_local = threading.local()
        self._exit_stack = exit_stack

    def __call__(self):
        try:
            session = self._thread_local.session
        except AttributeError:
            with self._lock: # ExitStack might not be thread-safe either
                session = self._exit_stack.enter_context(requests.Session())
            self._thread_local.session = session
        return session


def main():
    parser = DownloaderArgumentParser()
    parser.add_argument('--name', metavar='PAT[,PAT...]',
        help='download only models whose names match at least one of the specified patterns')
    parser.add_argument('--list', type=Path, metavar='FILE.LST',
        help='download only models whose names match at least one of the patterns in the specified file')
    parser.add_argument('--all',  action='store_true', help='download all available models')
    parser.add_argument('--print_all', action='store_true', help='print all available models')
    parser.add_argument('--precisions', metavar='PREC[,PREC...]',
                        help='download only models with the specified precisions (actual for DLDT networks)')
    parser.add_argument('-o', '--output_dir', type=Path, metavar='DIR',
        default=Path.cwd(), help='path where to save models')
    parser.add_argument('--cache_dir', type=Path, metavar='DIR',
        help='directory to use as a cache for downloaded files')
    parser.add_argument('--num_attempts', type=positive_int_arg, metavar='N', default=1,
        help='attempt each download up to N times')
    parser.add_argument('--progress_format', choices=('text', 'json'), default='text',
        help='which format to use for progress reporting')
    # unlike Model Converter, -jauto is not supported here, because CPU count has no
    # relation to the optimal number of concurrent downloads
    parser.add_argument('-j', '--jobs', type=positive_int_arg, metavar='N', default=1,
        help='how many downloads to perform concurrently')

    args = parser.parse_args()

    def make_reporter(context):
        return common.Reporter(context,
            enable_human_output=args.progress_format == 'text',
            enable_json_output=args.progress_format == 'json')

    reporter = make_reporter(common.DirectOutputContext())

    cache = NullCache() if args.cache_dir is None else DirCache(args.cache_dir)
    models = common.load_models_from_args(parser, args)

    failed_models = set()

    if args.precisions is None:
        requested_precisions = common.KNOWN_PRECISIONS
    else:
        requested_precisions = set(args.precisions.split(','))
        unknown_precisions = requested_precisions - common.KNOWN_PRECISIONS
        if unknown_precisions:
            sys.exit('Unknown precisions specified: {}.'.format(', '.join(sorted(unknown_precisions))))

    reporter.print_group_heading('Downloading models')
    with contextlib.ExitStack() as exit_stack:
        session_factory = ThreadSessionFactory(exit_stack)
        if args.jobs == 1:
            results = [download_model(reporter, args, cache, session_factory, requested_precisions, model)
                for model in models]
        else:
            results = common.run_in_parallel(args.jobs,
                lambda context, model: download_model(
                    make_reporter(context), args, cache, session_factory, requested_precisions, model),
                models)

    failed_models = {model.name for model, successful in zip(models, results) if not successful}

    reporter.print_group_heading('Post-processing')
    for model in models:
        if model.name in failed_models or not model.postprocessing: continue

        reporter.emit_event('model_postprocessing_begin', model=model.name)

        output = args.output_dir / model.subdirectory

        for postproc in model.postprocessing:
            postproc.apply(reporter, output)

        reporter.emit_event('model_postprocessing_end', model=model.name)

    if failed_models:
        reporter.print('FAILED:')
        for failed_model_name in failed_models:
            reporter.print(failed_model_name)
        sys.exit(1)

if __name__ == '__main__':
    main()
