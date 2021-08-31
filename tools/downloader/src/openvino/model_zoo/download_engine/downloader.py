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

import functools
import hashlib
import requests
import ssl
import time
import types

from openvino.model_zoo.download_engine import cache

DOWNLOAD_TIMEOUT = 5 * 60

class Downloader:
    def __init__(self, output_dir=None, cache_dir=None, num_attempts=1, timeout=DOWNLOAD_TIMEOUT):
        self.output_dir = output_dir
        self.cache = cache.NullCache() if cache_dir is None else cache.DirCache(cache_dir)
        self.num_attempts = num_attempts
        self.timeout = timeout

    def _process_download(self, reporter, chunk_iterable, size, progress, file):
        start_time = time.monotonic()
        start_size = progress.size

        try:
            for chunk in chunk_iterable:
                reporter.job_context.check_interrupted()

                if chunk:
                    duration = time.monotonic() - start_time
                    progress.size += len(chunk)
                    progress.hasher.update(chunk)

                    if duration != 0:
                        speed = int((progress.size - start_size) / (1024 * duration))
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

    def _try_download(self, reporter, file, start_download, size):
        progress = types.SimpleNamespace(size=0)

        for attempt in range(self.num_attempts):
            if attempt != 0:
                retry_delay = 10
                reporter.print("Will retry in {} seconds...", retry_delay, flush=True)
                time.sleep(retry_delay)

            try:
                reporter.job_context.check_interrupted()
                chunk_iterable, continue_offset = start_download(offset=progress.size, timeout=self.timeout)

                if continue_offset not in {0, progress.size}:
                    # Somehow we neither restarted nor continued from where we left off.
                    # Try to restart.
                    chunk_iterable, continue_offset = start_download(offset=0, timeout=self.timeout)
                    if continue_offset != 0:
                        reporter.log_error("Remote server refuses to send whole file, aborting")
                        return None

                if continue_offset == 0:
                    file.seek(0)
                    file.truncate()
                    progress.size = 0
                    progress.hasher = hashlib.sha256()

                self._process_download(reporter, chunk_iterable, size, progress, file)

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

    def _try_retrieve_from_cache(self, reporter, model_file, destination):
        try:
            if self.cache.has(model_file.sha256):
                reporter.job_context.check_interrupted()

                reporter.print_section_heading('Retrieving {} from the cache', destination)
                if not self.cache.get(model_file, destination, reporter):
                    reporter.print('Will retry from the original source.')
                    reporter.print()
                    return False
                reporter.print()
                return True
        except Exception:
            reporter.log_warning('Cache retrieval failed; falling back to downloading', exc_info=True)
            reporter.print()

        return False

    @staticmethod
    def _try_update_cache(reporter, cache, hash, source):
        try:
            cache.put(hash, source)
        except Exception:
            reporter.log_warning('Failed to update the cache', exc_info=True)

    def _try_retrieve(self, reporter, destination, model_file, start_download):
        destination.parent.mkdir(parents=True, exist_ok=True)

        if self._try_retrieve_from_cache(reporter, model_file, destination):
            return True

        reporter.print_section_heading('Downloading {}', destination)

        success = False

        with destination.open('w+b') as f:
            actual_hash = self._try_download(reporter, f, start_download, model_file.size)

        if actual_hash and cache.verify_hash(reporter, actual_hash, model_file.sha256, destination):
            self._try_update_cache(reporter, self.cache, model_file.sha256, destination)
            success = True

        reporter.print()
        return success

    def download_model(self, reporter, session_factory, requested_precisions, model, known_precisions):
        session = session_factory()

        reporter.print_group_heading('Downloading {}', model.name)

        reporter.emit_event('model_download_begin', model=model.name, num_files=len(model.files))

        output = self.output_dir / model.subdirectory
        output.mkdir(parents=True, exist_ok=True)

        for model_file in model.files:
            if len(model_file.name.parts) == 2:
                p = model_file.name.parts[0]
                if p in known_precisions and p not in requested_precisions:
                    continue

            model_file_reporter = reporter.with_event_context(model=model.name, model_file=model_file.name.as_posix())
            model_file_reporter.emit_event('model_file_download_begin', size=model_file.size)

            destination = output / model_file.name

            if not self._try_retrieve(model_file_reporter, destination, model_file,
                    functools.partial(model_file.source.start_download, session, cache.CHUNK_SIZE)):
                try:
                    destination.unlink()
                except FileNotFoundError:
                    pass

                model_file_reporter.emit_event('model_file_download_end', successful=False)
                reporter.emit_event('model_download_end', model=model.name, successful=False)
                return False

            model_file_reporter.emit_event('model_file_download_end', successful=True)

        reporter.emit_event('model_download_end', model=model.name, successful=True)

        if model.postprocessing:
            reporter.emit_event('model_postprocessing_begin', model=model.name)

            for postproc in model.postprocessing:
                postproc.apply(reporter, output)

            reporter.emit_event('model_postprocessing_end', model=model.name)

            reporter.print()

        return True
