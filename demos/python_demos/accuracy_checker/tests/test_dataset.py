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
import pytest
from accuracy_checker.config import ConfigError
from .conftest import mock_path_exists

from accuracy_checker.dataset import Dataset


class MockPreprocessor:
    @staticmethod
    def process(images):
        return images


class TestDataset:
    def test_missed_name_raises_config_error_exception(self):
        local_dataset = {
            'annotation': 'custom',
            'data_source': 'custom',
            'metrics': [{
                'type': 'fppi',
                'mr_rates': [0.0, 0.1]
            }],
        }

        with pytest.raises(ConfigError):
            Dataset(local_dataset, MockPreprocessor())

    def test_setting_custom_dataset_with_missed_annotation_raises_config_error_exception(self):
        local_dataset = {
            'name': 'custom',
            'data_source': 'custom',
            'metrics': [{'type': 'map'}]
        }
        with pytest.raises(ConfigError):
            Dataset(local_dataset, MockPreprocessor())

    def test_setting_custom_dataset_with_missed_data_source_raises_config_error_exception(self, mocker):
        mock_path_exists(mocker)
        local_dataset = {
            'name': 'custom',
            'annotation': 'custom',
            'metrics': [{'type': 'map'}]
        }
        with pytest.raises(ConfigError):
           Dataset(local_dataset, MockPreprocessor())
