# Copyright (c) 2021-2024 Intel Corporation
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

import re
import requests

from omz_tools.download_engine import base, validation


class FileSource(base.TaggedBase):
    RE_CONTENT_RANGE_VALUE = re.compile(r'bytes (\d+)-\d+/(?:\d+|\*)')

    types = {}

    @classmethod
    def deserialize(cls, source):
        if isinstance(source, str):
            source = {'$type': 'http', 'url': source}
        return super().deserialize(source)

    @classmethod
    def http_range_headers(cls, offset):
        if offset == 0:
            return {}

        return {
            'Accept-Encoding': 'identity',
            'Range': 'bytes={}-'.format(offset),
        }

    @classmethod
    def handle_http_response(cls, response, chunk_size):
        if response.status_code == requests.codes.partial_content:
            match = cls.RE_CONTENT_RANGE_VALUE.fullmatch(response.headers.get('Content-Range', ''))
            if not match:
                # invalid range reply; return a negative offset to make
                # the download logic restart the download.
                return None, -1

            return response.iter_content(chunk_size=chunk_size), int(match.group(1))

        # either we didn't ask for a range, or the server doesn't support ranges

        if 'Content-Range' in response.headers:
            # non-partial responses aren't supposed to have range information
            return None, -1

        return response.iter_content(chunk_size=chunk_size), 0


class FileSourceHttp(FileSource):
    def __init__(self, url):
        self.url = url

    @classmethod
    def deserialize(cls, source):
        return cls(validation.validate_string('"url"', source['url']))

    def start_download(self, session, chunk_size, offset, timeout, **kwargs):
        response = session.get(self.url, stream=True, timeout=timeout,
            headers=self.http_range_headers(offset))
        response.raise_for_status()

        return self.handle_http_response(response, chunk_size)

FileSource.types['http'] = FileSourceHttp

class FileSourceGoogleDrive(FileSource):
    def __init__(self, id):
        self.id = id

    @classmethod
    def deserialize(cls, source):
        return cls(validation.validate_string('"id"', source['id']))

    def start_download(self, session, chunk_size, offset, timeout, **kwargs):
        range_headers = self.http_range_headers(offset)
        URL = 'https://docs.google.com/uc?export=download'
        response = session.get(URL, params={'id': self.id}, headers=range_headers,
            stream=True, timeout=timeout)
        response.raise_for_status()

        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                params = {'id': self.id, 'confirm': value}
                response = session.get(URL, params=params, headers=range_headers,
                    stream=True, timeout=timeout)
                response.raise_for_status()

        return self.handle_http_response(response, chunk_size)

FileSource.types['google_drive'] = FileSourceGoogleDrive
