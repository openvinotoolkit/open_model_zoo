"""
Copyright (c) 2018-2024 Intel Corporation

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

from collections import defaultdict
from ...utils import read_json
from .loader import Loader, DictLoaderMixin


class JSONLoader(DictLoaderMixin, Loader):
    """
    Class for loading output from another tool in json format.
    """

    __provider__ = 'json'

    def load(self, identifiers=None, adapter=None, **kwargs):
        progress_reporter = kwargs.get('progress')
        detection_list = read_json(self._data_path)
        if progress_reporter:
            num_iters = len(identifiers) if identifiers else len(detection_list)
            progress_reporter.reset(num_iters)
        data = defaultdict(dict)
        idx = 0
        for detection in detection_list:
            if 'timestamp' in detection:
                idx = int(detection['timestamp']) // 1000000000
            if identifiers and idx >= len(identifiers):
                break
            identifier = identifiers[idx] if identifiers else idx
            idx += 1
            if adapter:
                detection = adapter.process(detection, [identifier], [{}])
            data[identifier] = detection
            if progress_reporter:
                progress_reporter.update(idx, 1)
        return data

    def __getitem__(self, item):
        return self.data[item]
