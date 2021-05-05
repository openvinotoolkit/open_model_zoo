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

from enum import Enum
import cv2
import numpy as np
import pytest

from accuracy_checker.config import ConfigError
from accuracy_checker.preprocessor import (
    Crop,
    Normalize,
    Preprocessor,
    Resize,
    Flip,
    BgrToRgb,
    CropRect,
    ExtendAroundRect,
    PointAligner,
    GeometricOperationMetadata
)
from accuracy_checker.preprocessor.ie_preprocessor import ie_preprocess_available, IEPreprocessor
from accuracy_checker.preprocessor.preprocessing_executor import PreprocessingExecutor
from accuracy_checker.preprocessor.resize import _OpenCVResizer
from accuracy_checker.data_readers import DataRepresentation


class TestResize:
    def test_default_resize(self, mocker):
        cv2_resize_mock = mocker.patch('accuracy_checker.preprocessor.geometric_transformations.cv2.resize')
        resize = Preprocessor.provide('resize', {'type': 'resize', 'size': 200})

        input_mock = mocker.Mock()
        input_mock.shape = (480, 640, 3)
        resize(DataRepresentation(input_mock))
        assert resize.dst_width == 200
        assert resize.dst_height == 200
        cv2_resize_mock.assert_called_once_with(
            input_mock, (200, 200), interpolation=_OpenCVResizer.supported_interpolations()['LINEAR']
        )

    def test_custom_resize(self, mocker):
        cv2_resize_mock = mocker.patch('accuracy_checker.preprocessor.geometric_transformations.cv2.resize')

        resize = Preprocessor.provide(
            'resize', {'type': 'resize', 'dst_width': 126, 'dst_height': 128, 'interpolation': 'CUBIC'}
        )

        input_mock = mocker.Mock()
        input_mock.shape = (480, 640, 3)
        resize(DataRepresentation(input_mock))

        assert resize.dst_width == 126
        assert resize.dst_height == 128
        cv2_resize_mock.assert_called_once_with(
            input_mock, (126, 128),
            interpolation=_OpenCVResizer.supported_interpolations()['CUBIC']
        )

    def test_resize_without_save_aspect_ratio(self):
        name = 'mock_preprocessor'
        config = {'type': 'resize', 'dst_width': 150, 'dst_height': 150}
        input_image = np.ones((100, 50, 3))
        resize = Preprocessor.provide('resize', config, name)

        result = resize(DataRepresentation(input_image)).data

        assert result.shape == (150, 150, 3)

    def test_resize_save_aspect_ratio_unknown_raise_config_error(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide(
                'resize', {'type': 'resize', 'dst_width': 100, 'dst_height': 150, 'aspect_ratio_scale': 'unknown'}
            )

    def test_resize_save_aspect_ratio_height(self):
        input_image = np.ones((100, 50, 3))
        resize = Preprocessor.provide('resize', {
            'type': 'resize', 'dst_width': 100, 'dst_height': 150,
            'interpolation': 'CUBIC', 'aspect_ratio_scale': 'height'
        })
        result = resize(DataRepresentation(input_image)).data

        assert result.shape == (300, 100, 3)

    def test_resize_save_aspect_ratio_width(self):
        input_image = np.ones((100, 50, 3))
        resize = Preprocessor.provide('resize', {
            'type': 'resize', 'dst_width': 150, 'dst_height': 150, 'aspect_ratio_scale': 'width'
        })
        result = resize(DataRepresentation(input_image)).data

        assert result.shape == (150, 75, 3)

    def test_resize_save_aspect_ratio_for_greater_dim(self):
        input_image = np.ones((100, 50, 3))
        resize = Preprocessor.provide('resize', {
            'type': 'resize',
            'dst_width': 100,
            'dst_height': 150,
            'aspect_ratio_scale': 'greater'
        })
        result = resize(DataRepresentation(input_image)).data

        assert result.shape == (300, 100, 3)

    def test_resize_save_aspect_ratio_frcnn_keep_aspect_ratio(self):
        input_image = np.ones((480, 640, 3))
        resize = Preprocessor.provide('resize', {
            'type': 'resize',
            'dst_width': 100,
            'dst_height': 150,
            'aspect_ratio_scale': 'frcnn_keep_aspect_ratio'
        })
        result = resize(DataRepresentation(input_image))

        assert result.data.shape == (100, 133, 3)
        assert result.metadata == {
                'geometric_operations': [
                    GeometricOperationMetadata(
                        type='resize',
                        parameters={
                            'scale_x': 0.2078125,
                            'scale_y': 0.20833333333333334,
                            'image_info': [100, 133, 1],
                            'original_width': 640,
                            'original_height': 480,
                            'preferable_width': 133,
                            'preferable_height': 150
                        }
                    )
                ],
                'image_info': [100, 133, 1],
                'image_size': (480, 640, 3),
                'original_height': 480,
                'original_width': 640,
                'preferable_height': 150,
                'preferable_width': 133,
                'scale_x': 0.2078125,
                'scale_y': 0.20833333333333334
        }

    def test_resize_to_negative_size_raise_config_error(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('resize', {'type': 'resize', 'size': -100})

    def test_resize_to_negative_destination_width_raise_config_error(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('resize', {'type': 'resize', 'dst_width': -100, 'dst_height': 100})

    def test_resize_to_negative_destination_height_raise_config_error(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('resize', {'type': 'resize', 'dst_width': 100, 'dst_height': -100})

    def test_resize_with_both_provided_size_and_dst_height_dst_width_warn(self):
        input_image = np.ones((100, 50, 3))

        with pytest.warns(None) as warnings:
            resize = Preprocessor.provide(
                'resize', {'type': 'resize', 'dst_width': 100, 'dst_height': 100, 'size': 200}
            )
            assert len(warnings) == 1
            result = resize(DataRepresentation(input_image)).data
            assert result.shape == (200, 200, 3)

    def test_resize_provided_only_dst_height_raise_config_error(self):
        with pytest.raises(ValueError):
            Preprocessor.provide('resize', {'type': 'resize', 'dst_height': 100})

    def test_resize_provided_only_dst_width_raise_config_error(self):
        with pytest.raises(ValueError):
            Preprocessor.provide('resize', {'type': 'resize', 'dst_width': 100})


class TestAutoResize:
    def test_default_auto_resize(self, mocker):
        cv2_resize_mock = mocker.patch('accuracy_checker.preprocessor.geometric_transformations.cv2.resize')
        resize = Preprocessor.provide('auto_resize', {'type': 'auto_resize'})
        resize.set_input_shape({'data': (1, 3, 200, 200)})

        input_data = np.zeros((100, 100, 3))
        input_rep = DataRepresentation(input_data)
        expected_meta = {
                    'preferable_width': 200,
                    'preferable_height': 200,
                    'image_info': [200, 200, 1],
                    'scale_x': 2.0,
                    'scale_y': 2.0,
                    'original_width': 100,
                    'original_height': 100,
                }
        resize(input_rep)

        assert resize.dst_width == 200
        assert resize.dst_height == 200
        cv2_resize_mock.assert_called_once_with(input_data, (200, 200))
        for key, value in expected_meta.items():
            assert key in input_rep.metadata
            assert input_rep.metadata[key] == value

    def test_auto_resize_input_shape_not_provided_raise_config_error(self, mocker):
        input_mock = mocker.Mock()
        with pytest.raises(ConfigError):
            Preprocessor.provide('auto_resize', {'type': 'auto_resize'})(DataRepresentation(input_mock))

    def test_auto_resize_with_non_image_input_raise_config_error(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('auto_resize', {'type': 'auto_resize'}).set_input_shape({'im_info': [200, 200, 1]})

    def test_auto_resize_empty_input_shapes_raise_config_error(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('auto_resize', {'type': 'auto_resize'}).set_input_shape({})


class TestNormalization:
    def test_normalization_without_mean_and_std_raise_config_error(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('normalization', {'type': 'normalization'})

    def test_custom_normalization_with_mean(self):
        normalization = Preprocessor.provide('normalization', {'type': 'normalization', 'mean': '(1, 2, 3)'})
        source = np.full_like((300, 300, 3), 100)
        input_ref = source.copy() - (1, 2, 3)
        result = normalization(DataRepresentation(source))

        assert normalization.mean == (1, 2, 3)
        assert normalization.std is None
        assert np.all(input_ref == result.data)
        assert result.metadata == {'image_size': (3,)}

    def test_custom_normalization_single_channel(self):
        normalization = Preprocessor.provide('normalization', {'type': 'normalization', 'mean': '1)'})
        source = np.full_like((300, 300), 100)
        input_ref = source.copy() - 1
        result = normalization(DataRepresentation(source))

        assert normalization.mean == (1, )
        assert normalization.std is None
        assert np.all(input_ref == result.data)
        assert result.metadata == {'image_size': (2,)}

    def test_custom_normalization_multi_input(self):
        normalization = Preprocessor.provide('normalization', {'type': 'normalization', 'mean': '2', 'std': '2'})
        source = np.full_like((300, 300, 3), 100)
        input_ref = (source.copy() - 2) / 2
        result = normalization(DataRepresentation([source, source]))

        assert normalization.mean == (2, )
        assert normalization.std == (2, )
        assert len(result.data) == 2
        assert np.all(input_ref == result.data[0])
        assert np.all(input_ref == result.data[1])
        assert result.metadata == {'image_size': (3,)}

    def test_custom_normalization_multi_input_images_only(self):
        normalization = Preprocessor.provide('normalization', {'type': 'normalization', 'mean': '2', 'std': '2', 'images_only': True})
        source = np.full((300, 300, 3), 100)
        input_ref = (source.copy() - 2) / 2
        result = normalization(DataRepresentation([source, 2]))

        assert normalization.mean == (2, )
        assert normalization.std == (2, )
        assert len(result.data) == 2
        assert np.all(input_ref == result.data[0])
        assert result.data[1] == 2
        assert result.metadata == {'image_size': (300, 300, 3)}

    def test_custom_normalization_with_precomputed_mean(self):
        normalization = Preprocessor.provide('normalization', {'type': 'normalization', 'mean': 'cifar10'})

        source = np.full_like((300, 300, 3), 100)
        input_ref = source.copy() - normalization.PRECOMPUTED_MEANS['cifar10']
        result = normalization(DataRepresentation(source))

        assert normalization.mean == normalization.PRECOMPUTED_MEANS['cifar10']
        assert normalization.std is None
        assert np.all(input_ref == result.data)
        assert result.metadata == {'image_size': (3,)}

    def test_custom_normalization_with_mean_as_scalar(self):
        normalization = Preprocessor.provide('normalization', {'type': 'normalization', 'mean': '1'})

        source = np.full_like((300, 300, 3), 100)
        input_ref = source.copy() - 1
        result = normalization(DataRepresentation(source))

        assert normalization.mean == (1.0, )
        assert normalization.std is None
        assert np.all(input_ref == result.data)
        assert result.metadata == {'image_size': (3,)}

    def test_custom_normalization_with_std(self):
        normalization = Preprocessor.provide('normalization', {'type': 'normalization', 'std': '(1, 2, 3)'})

        source = np.full_like((300, 300, 3), 100)
        input_ref = source.copy() / (1, 2, 3)
        result = normalization(DataRepresentation(source))

        assert normalization.mean is None
        assert normalization.std == (1, 2, 3)
        assert np.all(input_ref == result.data)
        assert result.metadata == {'image_size': (3,)}

    def test_custom_normalization_with_precomputed_std(self):
        normalization = Preprocessor.provide('normalization', {'type': 'normalization', 'std': 'cifar10'})

        source = np.full_like((300, 300, 3), 100)
        input_ref = source.copy() / normalization.PRECOMPUTED_STDS['cifar10']
        result = normalization(DataRepresentation(source))

        assert normalization.mean is None
        assert normalization.std == normalization.PRECOMPUTED_STDS['cifar10']
        assert np.all(input_ref == result.data)
        assert result.metadata == {'image_size': (3,)}

    def test_custom_normalization_with_std_as_scalar(self):
        normalization = Preprocessor.provide('normalization', {'type': 'normalization', 'std': '2'})
        source = np.full_like((300, 300, 3), 100)
        input_ref = source.copy() / 2
        result = normalization(DataRepresentation(source))

        assert normalization.mean is None
        assert normalization.std == (2.0, )
        assert np.all(input_ref == result.data)
        assert result.metadata == {'image_size': (3,)}

    def test_custom_normalization_with_mean_and_std(self):
        normalization = Preprocessor.provide(
            'normalization', {'type': 'normalization', 'mean': '(1, 2, 3)', 'std': '(4, 5, 6)'}
        )

        input_ = np.full_like((300, 300, 3), 100)
        input_ref = (input_ - (1, 2, 3)) / (4, 5, 6)
        result = normalization(DataRepresentation(input_))

        assert normalization.mean == (1, 2, 3)
        assert normalization.std == (4, 5, 6)
        assert np.all(input_ref == result.data)
        assert result.metadata == {'image_size': (3,)}

    def test_custom_normalization_with_mean_and_std_as_scalars(self):
        normalization = Preprocessor.provide('normalization', {'type': 'normalization', 'mean': '2', 'std': '5'})

        input_ = np.full_like((300, 300, 3), 100)
        input_ref = (input_ - (2, )) / (5, )
        result = normalization(DataRepresentation(input_))

        assert normalization.mean == (2, )
        assert normalization.std == (5, )
        assert np.all(input_ref == result.data)
        assert result.metadata == {'image_size': (3,)}

    def test_normalization_with_zero_in_std_values_raise_config_error(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('normalization', {'type': 'normalization', 'std': '(4, 0, 6)'})

    def test_normalization_with_zero_as_std_value_raise_config_error(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('normalization', {'type': 'normalization', 'std': '0'})

    def test_normalization_with_not_channel_wise_mean_list_raise_config_error(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('normalization', {'type': 'normalization', 'mean': '3, 2'})

    def test_normalization_with_not_channel_wise_std_list_raise_config_error(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('normalization', {'type': 'normalization', 'std': '3, 2'})

    def test_normalization_with_unknown_precomputed_mean_raise_config_error(self):
        with pytest.raises(ValueError):
            Preprocessor.provide('normalization', {'type': 'normalization', 'mean': 'unknown'})

    def test_normalization_with_unknown_precomputed_std_raise_config_error(self):
        with pytest.raises(ValueError):
            Preprocessor.provide('normalization', {'type': 'normalization', 'std': 'unknown'})


class TestPreprocessingEvaluator:
    def test_preprocessing_evaluator(self):
        config = [{'type': 'normalization', 'mean': '(1, 2, 3)'}, {'type': 'resize', 'size': 200}]
        preprocessor = PreprocessingExecutor(config)

        assert 2 == len(preprocessor.processors)
        assert isinstance(preprocessor.processors[0], Normalize)
        assert isinstance(preprocessor.processors[1], Resize)
        assert preprocessor.processors[0].mean == (1, 2, 3)
        assert preprocessor.processors[1].dst_width == 200


class TestCrop:
    def test_crop_higher(self):
        crop = Crop({'dst_width': 50, 'dst_height': 33, 'type': 'crop'})
        image = np.zeros((100, 100, 3))
        image_rep = crop(DataRepresentation(image))

        assert image_rep.data.shape == (33, 50, 3)
        assert image_rep.metadata == {'image_size': (100, 100, 3),
                'geometric_operations': [GeometricOperationMetadata(type='crop', parameters={})]}

    def test_crop_to_size(self):
        crop = Crop({'size': 50, 'type': 'crop'})
        image = np.zeros((100, 100, 3))
        image_rep = crop(DataRepresentation(image))

        assert image_rep.data.shape == (50, 50, 3)
        assert image_rep.metadata == {'image_size': (100, 100, 3),
                'geometric_operations': [GeometricOperationMetadata(type='crop', parameters={})]}

    def test_crop_higher_non_symmetric(self):
        crop = Crop({'dst_width': 50, 'dst_height': 12, 'type': 'crop'})
        image = np.zeros((70, 50, 3))
        image_rep = crop(DataRepresentation(image))

        assert image_rep.data.shape == (12, 50, 3)
        assert image_rep.metadata == {'image_size': (70, 50, 3),
                'geometric_operations': [GeometricOperationMetadata(type='crop', parameters={})]}

    def test_crop_less(self):
        crop = Crop({'dst_width': 151, 'dst_height': 42, 'type': 'crop'})
        image = np.zeros((30, 30, 3))
        image_rep = crop(DataRepresentation(image))

        assert image_rep.data.shape == (42, 151, 3)
        assert image_rep.metadata == {'image_size': (30, 30, 3),
                'geometric_operations': [GeometricOperationMetadata(type='crop', parameters={})]}

    def test_crop_less_non_symmetric(self):
        crop = Crop({'dst_width': 42, 'dst_height': 151, 'type': 'crop'})
        image = np.zeros((30, 40, 3))
        image_rep = crop(DataRepresentation(image))

        assert image_rep.data.shape == (151, 42, 3)
        assert image_rep.metadata == {'image_size': (30, 40, 3),
                'geometric_operations': [GeometricOperationMetadata(type='crop', parameters={})]}

    def test_crop_central_fraction_symmetric(self):
        crop = Crop({'central_fraction': 0.5, 'type': 'crop'})
        image = np.zeros((40, 40, 3))
        image_rep = crop(DataRepresentation(image))

        assert image_rep.data.shape == (20, 20, 3)
        assert image_rep.metadata == {'image_size': (40, 40, 3),
                'geometric_operations': [GeometricOperationMetadata(type='crop', parameters={})]}

    def test_crop_central_fraction_non_symmetric(self):
        crop = Crop({'central_fraction': 0.5, 'type': 'crop'})
        image = np.zeros((80, 40, 3))
        image_rep = crop(DataRepresentation(image))

        assert image_rep.data.shape == (40, 20, 3)
        assert image_rep.metadata == {'image_size': (80, 40, 3),
                'geometric_operations': [GeometricOperationMetadata(type='crop', parameters={})]}

    def test_crop_to_negative_size_raise_config_error(self):
        with pytest.raises(ConfigError):
            Crop({'size': -151, 'type': 'crop'})

    def test_crop_to_negative_destination_width_raise_config_error(self):
        with pytest.raises(ConfigError):
            Crop({'dst_width': -100, 'dst_height': 100, 'type': 'crop'})

    def test_crop_to_negative_destination_height_raise_config_error(self):
        with pytest.raises(ConfigError):
            Crop({'dst_width': 100, 'dst_height': -100, 'type': 'crop'})

    def test_crop_with_both_provided_size_and_dst_height_dst_width_warn(self):
        image = np.zeros((30, 40, 3))
        with pytest.warns(None) as warnings:
            crop = Crop({'dst_width': 100, 'dst_height': 100, 'size': 200, 'type': 'crop'})
            assert len(warnings) == 1
            result = crop.process(DataRepresentation(image))
            assert result.data.shape == (200, 200, 3)
            assert result.metadata == {'image_size': (30, 40, 3),
                    'geometric_operations': [GeometricOperationMetadata(type='crop', parameters={})]}

    def test_crop_not_provided_size_dst_height_dst_width_central_fraction_raise_config_error(self):
        with pytest.raises(ConfigError):
            Crop({'type': 'crop'})

    def test_crop_provided_size_and_central_fraction_raise_config_error(self):
        with pytest.raises(ConfigError):
            Crop({'type': 'crop', 'size': 200, 'central_fraction': 0.875})

    def test_crop_provided_dst_height_dst_width_and_central_fraction_raise_config_error(self):
        with pytest.raises(ConfigError):
            Crop({'type': 'crop', 'dst_height': 200, 'dst_width': 100, 'central_fraction': 0.875})

    def test_crop_with_negative_central_fraction_raise_config_error(self):
        with pytest.raises(ConfigError):
            Crop({'type': 'crop', 'central_fraction': -0.875})

    def test_crop_with_central_fraction_more_1_raise_config_error(self):
        with pytest.raises(ConfigError):
            Crop({'type': 'crop', 'central_fraction': 2})


class TestFlip:
    # TODO: check image metadata after flip?
    def test_horizontal_flip(self):
        image = np.random.randint(0, 255, (30, 40, 3))
        expected_image = cv2.flip(image, 0)
        flip = Flip({'type': 'flip', 'mode': 'horizontal'})
        assert np.array_equal(expected_image, flip.process(DataRepresentation(image)).data)

    def test_vertical_flip(self):
        image = np.random.randint(0, 255, (30, 40, 3))
        expected_image = cv2.flip(image, 1)
        flip = Flip({'type': 'flip', 'mode': 'vertical'})
        assert np.array_equal(expected_image, flip.process(DataRepresentation(image)).data)

    def test_flip_default_value_if_mode_not_provided(self):
        flip = Flip({'type': 'flip'})
        assert np.array_equal(0, flip.mode)

    def test_flip_raise_config_error_if_mode_unknown(self):
        with pytest.raises(ConfigError):
            Flip({'type': 'flip', 'mode': 'unknown'})


class TestBGRtoRGB:
    def test_bgr_to_rgb(self):
        image = np.random.randint(0, 255, (30, 40, 3)).astype(np.uint8)
        expected_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bgr_to_rgb = BgrToRgb({'type': 'bgr_to_rgb'})
        assert np.array_equal(expected_image, bgr_to_rgb.process(DataRepresentation(image)).data)


class TestCropRect:
    def test_crop_rect_if_rect_not_provided(self):
        image = np.zeros((30, 40, 3))
        crop_rect = CropRect({'type': 'crop_rect'})
        assert np.array_equal(image, crop_rect(image, {}))

    def test_crop_rect_if_rect_equal_image(self):
        image = np.zeros((30, 40, 3))
        crop_rect = CropRect({'type': 'crop_rect'})
        assert np.array_equal(image, crop_rect(DataRepresentation(image), {'rect': [0, 0, 40, 30]}).data)

    def test_crop_rect(self):
        image = np.zeros((30, 40, 3))
        image[:, 20:, :] = 1
        expected_image = np.ones((30, 20, 3))
        crop_rect = CropRect({'type': 'crop_rect'})
        assert np.array_equal(expected_image, crop_rect(DataRepresentation(image), {'rect': [20, 0, 40, 30]}).data)

    def test_crop_rect_negative_coordinates_of_rect(self):
        image = np.zeros((30, 40, 3))
        image[:, 20:, :] = 1
        expected_image = image
        crop_rect = CropRect({'type': 'crop_rect'})
        assert np.array_equal(expected_image, crop_rect(DataRepresentation(image), {'rect': [-20, 0, 40, 30]}).data)

    def test_crop_rect_more_image_size_coordinates_of_rect(self):
        image = np.zeros((30, 40, 3))
        image[:, 20:, :] = 1
        expected_image = np.ones((30, 20, 3))
        crop_rect = CropRect({'type': 'crop_rect'})
        assert np.array_equal(expected_image, crop_rect(DataRepresentation(image), {'rect': [20, 0, 40, 50]}).data)


class TestExtendAroundRect:
    def test_default_extend_around_rect_without_rect(self):
        image = np.random.randint(0, 255, (30, 40, 3)).astype(np.uint8)
        expected_image = image
        extend_image_around_rect = ExtendAroundRect({'type': 'extend_around_rect'})
        assert np.array_equal(expected_image, extend_image_around_rect(DataRepresentation(image), {}).data)

    def test_default_extend_around_rect(self):
        image = np.random.randint(0, 255, (30, 40, 3)).astype(np.uint8)
        expected_image = image
        extend_image_around_rect = ExtendAroundRect({'type': 'extend_around_rect'})
        assert np.array_equal(
            expected_image, extend_image_around_rect(DataRepresentation(image), {'rect': [20, 0, 40, 30]}).data
        )

    def test_extend_around_rect_with_positive_augmentation(self):
        image = np.random.randint(0, 255, (30, 40, 3)).astype(np.uint8)
        expected_image = cv2.copyMakeBorder(image, int(15.5), int(31), int(0), int(11), cv2.BORDER_REPLICATE)
        extend_image_around_rect = ExtendAroundRect({'type': 'extend_around_rect', 'augmentation_param': 0.5})
        assert np.array_equal(
            expected_image, extend_image_around_rect(DataRepresentation(image), {'rect': [20, 0, 40, 30]}).data
        )

    def test_extend_around_rect_with_negative_augmentation(self):
        image = np.random.randint(0, 255, (30, 40, 3)).astype(np.uint8)
        expected_image = image
        extend_image_around_rect = ExtendAroundRect({'type': 'extend_around_rect', 'augmentation_param': -0.5})
        assert np.array_equal(
            expected_image, extend_image_around_rect(DataRepresentation(image), {'rect': [20, 0, 40, 30]}).data
        )

    def test_extend_around_rect_with_rect_equal_image(self):
        image = np.random.randint(0, 255, (30, 40, 3)).astype(np.uint8)
        expected_image = cv2.copyMakeBorder(image, int(15.5), int(31), int(20.5), int(41), cv2.BORDER_REPLICATE)
        extend_image_around_rect = ExtendAroundRect({'type': 'extend_around_rect', 'augmentation_param': 0.5})
        assert np.array_equal(
            expected_image, extend_image_around_rect(DataRepresentation(image), {'rect': [0, 0, 40, 30]}).data
        )

    def test_extend_around_rect_negative_coordinates_of_rect(self):
        image = np.random.randint(0, 255, (30, 40, 3)).astype(np.uint8)
        expected_image = cv2.copyMakeBorder(image, int(15.5), int(31), int(20.5), int(41), cv2.BORDER_REPLICATE)
        extend_image_around_rect = ExtendAroundRect({'type': 'extend_around_rect', 'augmentation_param': 0.5})
        assert np.array_equal(
            expected_image, extend_image_around_rect(DataRepresentation(image), {'rect': [-20, 0, 40, 30]}).data
        )

    def test_extend_around_rect_more_image_size_coordinates_of_rect(self):
        image = np.random.randint(0, 255, (30, 40, 3)).astype(np.uint8)
        expected_image = cv2.copyMakeBorder(image, int(15.5), int(31), int(0), int(11), cv2.BORDER_REPLICATE)
        extend_image_around_rect = ExtendAroundRect({'type': 'extend_around_rect', 'augmentation_param': 0.5})
        assert np.array_equal(
            expected_image, extend_image_around_rect(DataRepresentation(image), {'rect': [20, 0, 40, 50]}).data
        )


class TestPointAlignment:
    def test_point_alignment_width_negative_size_raise_config_error(self):
        with pytest.raises(ConfigError):
            PointAligner({'type': 'point_alignment', 'size': -100})

    def test_point_alignment_negative_destination_width_raise_config_error(self):
        with pytest.raises(ConfigError):
            PointAligner({'type': 'point_alignment', 'dst_width': -100, 'dst_height': 100})

    def test_point_alignment_to_negative_destination_height_raise_config_error(self):
        with pytest.raises(ValueError):
            PointAligner({'type': 'point_alignment', 'dst_width': 100, 'dst_height': -100})

    def test_point_alignment_provided_only_dst_height_raise_config_error(self):
        with pytest.raises(ValueError):
            PointAligner({'type': 'point_alignment', 'dst_height': 100})

    def test_point_alignment_provided_only_dst_width_raise_config_error(self):
        with pytest.raises(ValueError):
            PointAligner({'type': 'point_alignment', 'dst_width': 100})

    def test_point_alignment_both_provided_size_and_dst_height_dst_width_warn(self):
        input_image = np.ones((100, 50, 3))

        with pytest.warns(None) as warnings:
            point_aligner = PointAligner({'type': 'point_alignment', 'dst_width': 100, 'dst_height': 100, 'size': 200})
            assert len(warnings) == 1
            result = point_aligner(DataRepresentation(input_image), {}).data
            assert result.shape == (100, 50, 3)

    def test_point_alignment_not_provided_points_im_meta(self):
        input_image = np.ones((100, 50, 3))

        point_aligner = PointAligner({'type': 'point_alignment', 'dst_width': 100, 'dst_height': 100})
        result = point_aligner(DataRepresentation(input_image), {}).data
        assert result.shape == (100, 50, 3)

    def test_point_alignment_default_use_normalization(self):
        image = np.random.randint(0, 255, (40, 40, 3)).astype(np.uint8)

        point_aligner = PointAligner({'type': 'point_alignment', 'dst_width': 40, 'dst_height': 40})
        result = point_aligner(
            DataRepresentation(image), {'keypoints': PointAligner.ref_landmarks.reshape(-1).tolist()}
        ).data
        transformation_matrix = point_aligner.transformation_from_points(
            point_aligner.ref_landmarks * 40, point_aligner.ref_landmarks
        )
        expected_result = cv2.warpAffine(image, transformation_matrix, (40, 40), flags=cv2.WARP_INVERSE_MAP)

        assert np.array_equal(result, expected_result)

    def test_point_alignment_use_normalization(self):
        image = np.random.randint(0, 255, (40, 40, 3)).astype(np.uint8)

        point_aligner = PointAligner({'type': 'point_alignment', 'dst_width': 40, 'dst_height': 40, 'normalize': True})
        result = point_aligner(
            DataRepresentation(image), {'keypoints': PointAligner.ref_landmarks.reshape(-1).tolist()}
        ).data
        transformation_matrix = point_aligner.transformation_from_points(
            point_aligner.ref_landmarks * 40, point_aligner.ref_landmarks
        )
        expected_result = cv2.warpAffine(image, transformation_matrix, (40, 40), flags=cv2.WARP_INVERSE_MAP)

        assert np.array_equal(result, expected_result)

    def test_point_alignment_without_normalization(self):
        image = np.random.randint(0, 255, (40, 40, 3)).astype(np.uint8)

        point_aligner = PointAligner({'type': 'point_alignment', 'dst_width': 40, 'dst_height': 40, 'normalize': False})
        result = point_aligner(
            DataRepresentation(image), {'keypoints': PointAligner.ref_landmarks.reshape(-1).tolist()}
        ).data
        transformation_matrix = point_aligner.transformation_from_points(
            point_aligner.ref_landmarks * 40, point_aligner.ref_landmarks * 40
        )
        expected_result = cv2.warpAffine(image, transformation_matrix, (40, 40), flags=cv2.WARP_INVERSE_MAP)

        assert np.array_equal(result, expected_result)

    def test_point_alignment_with_drawing_points(self):
        image = np.random.randint(0, 255, (40, 40, 3)).astype(np.uint8)

        point_aligner = PointAligner({
            'type': 'point_alignment', 'dst_width': 40, 'dst_height': 40, 'draw_points': True
        })
        result = point_aligner(
            DataRepresentation(image), {'keypoints': PointAligner.ref_landmarks.reshape(-1).tolist()}
        ).data
        transformation_matrix = point_aligner.transformation_from_points(
            point_aligner.ref_landmarks * 40, point_aligner.ref_landmarks
        )
        expected_result = image
        for point in PointAligner.ref_landmarks:
            cv2.circle(expected_result, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)
        expected_result = cv2.warpAffine(expected_result, transformation_matrix, (40, 40), flags=cv2.WARP_INVERSE_MAP)

        assert np.array_equal(result, expected_result)

    def test_point_alignment_with_resizing(self):
        image = np.random.randint(0, 255, (80, 80, 3)).astype(np.uint8)

        point_aligner = PointAligner({'type': 'point_alignment', 'size': 40})
        result = point_aligner(
            DataRepresentation(image), {'keypoints': PointAligner.ref_landmarks.reshape(-1).tolist()}
        ).data
        transformation_matrix = point_aligner.transformation_from_points(
            point_aligner.ref_landmarks * 40, point_aligner.ref_landmarks * 0.5
        )
        expected_result = cv2.resize(image, (40, 40))
        expected_result = cv2.warpAffine(expected_result, transformation_matrix, (40, 40), flags=cv2.WARP_INVERSE_MAP)

        assert np.array_equal(result, expected_result)


class TestPreprocessorExtraArgs:
    def test_resize_raise_config_error_on_extra_args(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('resize', {'type': 'resize', 'size': 1, 'something_extra': 'extra'})

    def test_auto_resize_raise_config_error_on_extra_args(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('auto_resize', {'type': 'auto_resize', 'something_extra': 'extra'},)

    def test_normalization_raise_config_error_on_extra_args(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('normalization', {'type': 'normalization', 'mean': 0, 'something_extra': 'extra'})

    def test_bgr_to_rgb_raise_config_error_on_extra_args(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('bgr_to_rgb', {'type': 'bgr_to_rgb', 'something_extra': 'extra'})

    def test_flip_raise_config_error_on_extra_args(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('flip', {'type': 'flip', 'something_extra': 'extra'})

    def test_crop_accuracy_raise_config_error_on_extra_args(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('crop', {'type': 'crop', 'size': 1, 'something_extra': 'extra'})

    def test_extend_around_rect_raise_config_error_on_extra_args(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('extend_around_rect', {'type': 'extend_around_rect', 'something_extra': 'extra'})

    def test_point_alignment_raise_config_error_on_extra_args(self):
        with pytest.raises(ConfigError):
            Preprocessor.provide('point_alignment', {'type': 'point_alignment', 'something_extra': 'extra'})


@pytest.mark.skipif(not ie_preprocess_available(), reason='IE version does not support preprocessing')
class TestIEPreprocessor:
    def test_warn_on_no_supported_ops(self):
        config = [{'type': 'crop'}]
        with pytest.warns(UserWarning):
            preprocessor = IEPreprocessor(config)
            assert not preprocessor.steps

    def test_aspect_ratio_resize(self):
        config = [{'type': 'resize', 'aspect_ratio': 'higher'}]
        with pytest.warns(UserWarning):
            preprocessor = IEPreprocessor(config)
            assert not preprocessor.steps
            assert preprocessor.keep_preprocessing_info == config

    def test_unsupported_interpolation_resize(self):
        config = [{'type': 'resize', 'interpolation': 'unknown'}]
        with pytest.warns(UserWarning):
            preprocessor = IEPreprocessor(config)
            assert not preprocessor.steps
            assert preprocessor.keep_preprocessing_info == config

    def test_resize_no_interpolation_specified(self):
        config = [{'type': 'resize'}]
        preprocessor = IEPreprocessor(config)
        assert preprocessor.has_resize()
        assert len(preprocessor.steps) == 1
        assert preprocessor.steps[0].name == 'resize_algorithm'
        assert isinstance(preprocessor.steps[0].value, Enum)
        resize_interpolation = preprocessor.steps[0].value
        assert resize_interpolation.name == 'RESIZE_BILINEAR'
        assert not preprocessor.keep_preprocessing_info

    def test_resize_bilinear_interpolation_specified(self):
        config = [{'type': 'resize', 'interpolation': 'bilinear'}]
        preprocessor = IEPreprocessor(config)
        assert preprocessor.has_resize()
        assert len(preprocessor.steps) == 1
        assert preprocessor.steps[0].name == 'resize_algorithm'
        assert isinstance(preprocessor.steps[0].value, Enum)
        resize_interpolation = preprocessor.steps[0].value
        assert resize_interpolation.name == 'RESIZE_BILINEAR'
        assert not preprocessor.keep_preprocessing_info

    def test_resize_area_interpolation_specified(self):
        config = [{'type': 'resize', 'interpolation': 'area'}]
        preprocessor = IEPreprocessor(config)
        assert preprocessor.has_resize()
        assert len(preprocessor.steps) == 1
        assert preprocessor.steps[0].name == 'resize_algorithm'
        assert isinstance(preprocessor.steps[0].value, Enum)
        resize_interpolation = preprocessor.steps[0].value
        assert resize_interpolation.name == 'RESIZE_AREA'
        assert not preprocessor.keep_preprocessing_info

    def test_particular_preprocessing_transition(self):
        config = [{'type': 'crop'}, {'type': 'resize'}]
        preprocessor = IEPreprocessor(config)
        assert preprocessor.has_resize()
        assert len(preprocessor.steps) == 1
        assert preprocessor.steps[0].name == 'resize_algorithm'
        assert isinstance(preprocessor.steps[0].value, Enum)
        resize_interpolation = preprocessor.steps[0].value
        assert resize_interpolation.name == 'RESIZE_BILINEAR'
        assert len(preprocessor.keep_preprocessing_info) == 1
        assert preprocessor.keep_preprocessing_info[0] == config[0]

    def test_no_transit_preprocessing_if_last_operation_is_not_supported(self):
        config = [{'type': 'resize'}, {'type': 'crop'}]
        with pytest.warns(UserWarning):
            preprocessor = IEPreprocessor(config)
            assert not preprocessor.has_resize()
            assert not preprocessor.steps
            assert preprocessor.keep_preprocessing_info == config

    def test_unsupported_color_format(self):
        config = [{'type': 'bgr_to_gray'}]
        with pytest.warns(UserWarning):
            preprocessor = IEPreprocessor(config)
            assert not preprocessor.has_resize()
            assert not preprocessor.steps
            assert preprocessor.keep_preprocessing_info == config

    def test_rgb_color_format(self):
        config = [{'type': 'rgb_to_bgr'}]
        preprocessor = IEPreprocessor(config)
        assert not preprocessor.has_resize()
        assert len(preprocessor.steps) == 1
        assert not preprocessor.keep_preprocessing_info
        assert preprocessor.steps[0].name == 'color_format'
        color_space = preprocessor.steps[0].value
        assert isinstance(color_space, Enum)
        assert color_space.name == 'RGB'

    def test_nv12_color_format(self):
        config = [{'type': 'nv12_to_bgr'}]
        preprocessor = IEPreprocessor(config)
        assert not preprocessor.has_resize()
        assert len(preprocessor.steps) == 1
        assert not preprocessor.keep_preprocessing_info
        assert preprocessor.steps[0].name == 'color_format'
        color_space = preprocessor.steps[0].value
        assert isinstance(color_space, Enum)
        assert color_space.name == 'NV12'

    def test_partial_color_format(self):
        config = [{'type': 'bgr_to_nv12'}, {'type': 'nv12_to_bgr'}]
        preprocessor = IEPreprocessor(config)
        assert not preprocessor.has_resize()
        assert len(preprocessor.steps) == 1
        assert preprocessor.keep_preprocessing_info == [config[0]]
        assert preprocessor.steps[0].name == 'color_format'
        color_space = preprocessor.steps[0].value
        assert isinstance(color_space, Enum)
        assert color_space.name == 'NV12'

    def test_several_supported_preprocessing_ops(self):
        config = [{'type': 'resize'}, {'type': 'bgr_to_rgb'}]
        preprocessor = IEPreprocessor(config)
        assert preprocessor.has_resize()
        assert len(preprocessor.steps) == 2
        assert not preprocessor.keep_preprocessing_info
        assert preprocessor.steps[0].name == 'color_format'
        color_space = preprocessor.steps[0].value
        assert isinstance(color_space, Enum)
        assert color_space.name == 'BGR'
        assert preprocessor.steps[1].name == 'resize_algorithm'
        resize_interpolation = preprocessor.steps[1].value
        assert isinstance(resize_interpolation, Enum)
        assert resize_interpolation.name == 'RESIZE_BILINEAR'

    def test_mean_values_only(self):
        config = [{'type': 'normalization', 'mean': 255}]
        preprocessor = IEPreprocessor(config)
        assert preprocessor.has_normalization()
        assert len(preprocessor.steps) == 1
        assert preprocessor.steps[0].name == 'mean_variant'
        assert preprocessor.steps[0].value.name == 'MEAN_VALUE'
        assert preprocessor.mean_values == (255, )
        assert preprocessor.std_values is None

    def test_std_values_only(self):
        config = [{'type': 'normalization', 'std': 255}]
        preprocessor = IEPreprocessor(config)
        assert preprocessor.has_normalization()
        assert len(preprocessor.steps) == 1
        assert preprocessor.steps[0].name == 'mean_variant'
        assert preprocessor.steps[0].value.name == 'MEAN_VALUE'
        assert preprocessor.std_values == (255, )
        assert preprocessor.mean_values is None

    def test_mean_and_std_values(self):
        config = [{'type': 'normalization', 'mean': 255, 'std': 255}]
        preprocessor = IEPreprocessor(config)
        assert preprocessor.has_normalization()
        assert len(preprocessor.steps) == 1
        assert preprocessor.steps[0].name == 'mean_variant'
        assert preprocessor.steps[0].value.name == 'MEAN_VALUE'
        assert preprocessor.std_values == (255, )
        assert preprocessor.mean_values == (255, )

    def test_mean_values_for_each_channel(self):
        config = [{'type': 'normalization', 'mean': [255, 255, 255], 'std': [255, 255, 255]}]
        preprocessor = IEPreprocessor(config)
        assert preprocessor.has_normalization()
        assert len(preprocessor.steps) == 1
        assert preprocessor.steps[0].name == 'mean_variant'
        assert preprocessor.steps[0].value.name == 'MEAN_VALUE'
        assert preprocessor.std_values == [255, 255, 255]
        assert preprocessor.mean_values == [255, 255, 255]

    def test_precomputed_mean_values(self):
        config = [{'type': 'normalization', 'mean': 'imagenet', 'std': 255}]
        preprocessor = IEPreprocessor(config)
        assert preprocessor.has_normalization()
        assert len(preprocessor.steps) == 1
        assert preprocessor.steps[0].name == 'mean_variant'
        assert preprocessor.steps[0].value.name == 'MEAN_VALUE'
        assert preprocessor.std_values == (255, )
        assert preprocessor.mean_values == Normalize.PRECOMPUTED_MEANS['imagenet']
