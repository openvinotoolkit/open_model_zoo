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
import numpy as np
from accuracy_checker.preprocessor import Crop, Normalize, Preprocessor, Resize
from accuracy_checker.preprocessor.preprocessing_executor import PreprocessingExecutor


class TestResize:
    def test_default_resize(self, mocker):
        cv2_resize_mock = mocker.patch('accuracy_checker.preprocessor.preprocessors.cv2.resize')
        name = 'mock_preprocessor'
        config = {'type': 'resize', 'size': 200}
        resize = Preprocessor.provide('resize', config, name)

        input_mock = mocker.Mock()
        resize(input_mock)

        assert not resize.use_pil
        assert resize.dst_width == 200
        assert resize.dst_height == 200
        cv2_resize_mock.assert_called_once_with(input_mock, (200, 200),
                                                interpolation=Resize.OPENCV_INTERPOLATION['LINEAR'])

    def test_custom_resize(self, mocker):
        cv2_resize_mock = mocker.patch('accuracy_checker.preprocessor.preprocessors.cv2.resize')
        name = 'mock_preprocessor'
        config = {
            'type': 'resize',
            'dst_width': 126,
            'dst_height': 128,
            'interpolation': 'CUBIC'
        }

        resize = Preprocessor.provide('resize', config, name)

        input_mock = mocker.Mock()
        resize(input_mock)

        assert not resize.use_pil
        assert resize.dst_width == 126
        assert resize.dst_height == 128
        cv2_resize_mock.assert_called_once_with(input_mock, (126, 128),
                                                interpolation=Resize.OPENCV_INTERPOLATION['CUBIC'])


class TestNormalization:
    def test_default_normalization(self):
        name = 'mock_preprocessor'
        config = {'type': 'normalization'}
        normalization = Preprocessor.provide('normalization', config, name)

        input = np.full_like((3, 300, 300), 100)
        input_copy = input.copy()
        res = normalization(input)

        assert normalization.mean is None
        assert normalization.std is None
        assert np.all(input_copy == res)

    def test_custom_normalization_with_mean(self):
        name = 'mock_preprocessor'
        config = {'type': 'normalization', 'mean': '(1, 2, 3)'}
        normalization = Preprocessor.provide('normalization', config, name)

        input = np.full_like((3, 300, 300), 100)
        input_ref = input.copy() - (1, 2, 3)
        res = normalization(input)

        assert normalization.mean == (1, 2, 3)
        assert normalization.std is None
        assert np.all(input_ref == res)

    def test_custom_normalization_with_std(self):
        name = 'mock_preprocessor'
        config = {'type': 'normalization', 'std': '(1, 2, 3)'}
        normalization = Preprocessor.provide('normalization', config, name)

        input = np.full_like((3, 300, 300), 100)
        input_ref = input.copy() / (1, 2, 3)
        res = normalization(input)

        assert normalization.mean is None
        assert normalization.std == (1, 2, 3)
        assert np.all(input_ref == res)

    def test_custom_normalization_with_mean_and_std(self):
        name = 'mock_preprocessor'
        config = {'type': 'normalization', 'mean': '(1, 2, 3)', 'std': '(4, 5, 6)'}
        normalization = Preprocessor.provide('normalization', config, name)

        input_ = np.full_like((3, 300, 300), 100)
        input_ref = (input_ - (1, 2, 3)) / (4, 5, 6)
        res = normalization(input_)

        assert normalization.mean == (1, 2, 3)
        assert normalization.std == (4, 5, 6)
        assert np.all(input_ref == res)


def test_preprocessing_evaluator():
    config = [{'type': 'normalization', 'mean': '(1, 2, 3)'}, {'type': 'resize', 'size': 200}]
    preprocessor = PreprocessingExecutor(config)

    assert 2 == len(preprocessor.processors)
    assert isinstance(preprocessor.processors[0], Normalize)
    assert isinstance(preprocessor.processors[1], Resize)
    assert preprocessor.processors[0].mean == (1, 2, 3)
    assert preprocessor.processors[1].dst_width == 200


def test_crop__higher():
    crop = Crop({'dst_width': 50, 'dst_height': 33, 'type': 'crop'})
    image = np.zeros((100, 100, 3))
    image = crop(image)

    assert image.shape == (33, 50, 3)


def test_crop__higher_non_symmetric():
    crop = Crop({'dst_width': 50, 'dst_height': 12, 'type': 'crop'})
    image = np.zeros((70, 50, 3))
    image = crop(image)

    assert image.shape == (12, 50, 3)


def test_crop__less():
    crop = Crop({'dst_width': 151, 'dst_height': 42, 'type': 'crop'})
    image = np.zeros((30, 30, 3))
    image = crop(image)

    assert image.shape == (42, 151, 3)


def test_crop__less_non_symmetric():
    crop = Crop({'dst_width': 42, 'dst_height': 151, 'type': 'crop'})
    image = np.zeros((30, 40, 3))
    image = crop(image)

    assert image.shape == (151, 42, 3)
