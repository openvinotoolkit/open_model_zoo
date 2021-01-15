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

import pytest
import numpy as np
from accuracy_checker.launcher import DummyLauncher
from accuracy_checker.launcher.loaders import StoredPredictionBatch
from accuracy_checker.adapters import ClassificationAdapter
from accuracy_checker.representation import ClassificationPrediction


@pytest.mark.usefixtures('mock_file_exists')
class TestDummyLauncher:
    def test_empty_predictions_loading(self, mocker):
        launcher_config = {
            'framework': 'dummy',
            'loader': 'pickle',
            'data_path': '/path'
        }
        mocker.patch('accuracy_checker.launcher.loaders.pickle_loader.PickleLoader.read_pickle', return_value=[])
        launcher = DummyLauncher(launcher_config)
        assert not launcher._loader.data

    def test_access_to_non_existing_index(self, mocker):
        launcher_config = {
            'framework': 'dummy',
            'loader': 'pickle',
            'data_path': '/path'
        }
        mocker.patch('accuracy_checker.launcher.loaders.pickle_loader.PickleLoader.read_pickle', return_value=[])
        launcher = DummyLauncher(launcher_config)
        assert not launcher._loader.data
        with pytest.raises(IndexError):
            launcher.predict([1])

    def test_predictions_loading_without_adapter(self, mocker):
        launcher_config = {
            'framework': 'dummy',
            'loader': 'pickle',
            'data_path': '/path'
        }
        raw_prediction_batch = StoredPredictionBatch({'prediction': np.array([[0, 1]])}, [1], [{}])
        mocker.patch(
            'accuracy_checker.launcher.loaders.pickle_loader.PickleLoader.read_pickle',
            return_value=[raw_prediction_batch])
        launcher = DummyLauncher(launcher_config)
        assert len(launcher._loader.data) == 1
        assert launcher.predict([1]) == [raw_prediction_batch]

    def test_predictions_loading_with_adapter(self, mocker):
        launcher_config = {
            'framework': 'dummy',
            'loader': 'pickle',
            'data_path': '/path'
        }
        raw_prediction_batch = StoredPredictionBatch(
            {'prediction': np.array([[0, 1]])}, [1], [{}]
        )
        expected_prediction = ClassificationPrediction(1, np.array([0, 1]))
        adapter = ClassificationAdapter({'type': 'classification'})
        mocker.patch(
            'accuracy_checker.launcher.loaders.pickle_loader.PickleLoader.read_pickle',
            return_value=[raw_prediction_batch])
        launcher = DummyLauncher(launcher_config, adapter=adapter)
        assert len(launcher._loader.data) == 1
        prediction = launcher.predict([1])
        assert len(prediction) == 1
        assert isinstance(prediction[0], ClassificationPrediction)
        assert prediction[0].identifier == expected_prediction.identifier
        assert np.array_equal(prediction[0].scores, expected_prediction.scores)
