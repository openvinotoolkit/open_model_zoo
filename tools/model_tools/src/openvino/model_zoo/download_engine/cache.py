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

import hashlib
import shutil
import sys
import tempfile

from pathlib import Path

CHUNK_SIZE = 1 << 15 if sys.stdout.isatty() else 1 << 20


class NullCache:
    def has(self, hash): return False
    def get(self, model_file, path, reporter): return False
    def put(self, hash, path): pass


class DirCache:
    _FORMAT = 1 # increment if backwards-incompatible changes to the format are made
    _HASH_LEN = hashlib.sha384().digest_size

    def __init__(self, cache_dir):
        self._cache_dir = cache_dir / str(self._FORMAT)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._staging_dir = self._cache_dir / 'staging'
        self._staging_dir.mkdir(exist_ok=True)

    def _hash_path(self, hash):
        assert len(hash) == self._HASH_LEN
        hash_str = hash.hex().lower()
        return self._cache_dir / hash_str[:2] / hash_str[2:]

    def has(self, hash):
        return self._hash_path(hash).exists()

    def get(self, model_file, path, reporter):
        cache_path = self._hash_path(model_file.sha384)
        cache_sha384 = hashlib.sha384()
        cache_size = 0

        with open(cache_path, 'rb') as cache_file, open(path, 'wb') as destination_file:
            while True:
                data = cache_file.read(CHUNK_SIZE)
                if not data:
                    break
                cache_size += len(data)
                if cache_size > model_file.size:
                    reporter.log_error("Cached file is longer than expected ({} B), copying aborted", model_file.size)
                    return False
                cache_sha384.update(data)
                destination_file.write(data)
        if cache_size < model_file.size:
            reporter.log_error("Cached file is shorter ({} B) than expected ({} B)", cache_size, model_file.size)
            return False
        return verify_hash(reporter, cache_sha384.digest(), model_file.sha384, path)

    def put(self, hash, path):
        staging_path = None

        try:
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
            staging_path = None
        finally:
            # If we failed to complete our temporary file or to move it into place,
            # get rid of it.
            if staging_path:
                staging_path.unlink()


def verify_hash(reporter, actual_hash, expected_hash, path):
    if actual_hash != expected_hash:
        reporter.log_error('Hash mismatch for "{}"', path)
        reporter.log_details('Expected: {}', expected_hash.hex())
        reporter.log_details('Actual:   {}', actual_hash.hex())
        return False
    return True
