"""
Copyright (c) 2018-2021 Intel Corporation

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

import pickle
from .loader import Loader, DictLoaderMixin, StoredPredictionBatch


class PickleLoader(DictLoaderMixin, Loader):
    """
    Class for loading output from another tool in .pickle format.
    """

    __provider__ = 'pickle'

    def load(self, *args, **kwargs):
        progress_reporter = kwargs.get('progress')
        data = self.read_pickle(self._data_path)

        if isinstance(data, list):
            if progress_reporter:
                progress_reporter.reset(len(data))
            if all(isinstance(entry, StoredPredictionBatch) for entry in data):
                return self.load_batched_predictions(data, kwargs.get('adapter'), progress_reporter)

            if all(hasattr(entry, 'identifier') for entry in data):
                predictions = {}
                for idx, rep in enumerate(data):
                    predictions[rep.identifier] = rep
                    if progress_reporter:
                        progress_reporter.update(idx, 1)
                return predictions
            if 'identifiers' in kwargs:
                identifiers = kwargs['identifiers']
                return dict(zip(identifiers, data))

        return data

    @staticmethod
    def read_pickle(data_path):
        result = []
        with open(data_path, 'rb') as content:
            while True:
                try:
                    result.append(pickle.load(content))
                except EOFError:
                    break
        return result

    @staticmethod
    def load_batched_predictions(data, adapter=None, progress_reporter=None):
        predictions = {}
        for idx, entry in enumerate(data):
            if adapter:
                pred_list = adapter.process(*entry)
                for pred in pred_list:
                    predictions[pred.identifier] = pred
            else:
                for identifier in entry.identifiers:
                    predictions[identifier] = entry
            if progress_reporter:
                progress_reporter.update(idx, 1)
        return predictions
