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

import numpy as np
import pytest

from .common import make_segmentation_representation, make_instance_segmentation_representation
from accuracy_checker.utils import UnsupportedPackage

try:
    import pycocotools.mask as maskUtils
except ImportError as import_error:
    maskUtils = UnsupportedPackage("pycocotools", import_error.msg)

def no_available_pycocotools():
    return isinstance(maskUtils, UnsupportedPackage)

def encode_mask(mask):
    raw_mask = []
    for elem in mask:
        raw_mask.append(maskUtils.encode(np.asfortranarray(np.uint8(elem))))
    return raw_mask


class TestSegmentationRepresentation:
    def test_to_polygon_annotation(self):
        annotation = make_segmentation_representation(np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]), True)[0]
        expected = {
            0: [np.array([[1, 0], [3, 0], [3, 2]])],
            1: [np.array([[0, 0], [0, 2], [2, 2]])]}

        actual = annotation.to_polygon()

        for key in expected.keys():
            assert actual[key]
            for actual_arr, expected_arr in zip(actual[key], expected[key]):
                assert np.array_equal(actual_arr.sort(axis=0), expected_arr.sort(axis=0))

    def test_to_polygon_annotation_with_colors_in_arg(self):
        annotation = make_segmentation_representation(np.array(
            [[[128, 128, 128], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
             [[128, 128, 128], [128, 128, 128], [0, 0, 0], [0, 0, 0]],
             [[128, 128, 128], [128, 128, 128], [128, 128, 128], [0, 0, 0]]]), True)[0]
        segmentation_colors = [[0, 0, 0], [128, 128, 128]]
        expected = {
            0: [np.array([[1, 0], [3, 0], [3, 2]])],
            1: [np.array([[0, 0], [0, 2], [2, 2]])]}

        actual = annotation.to_polygon(segmentation_colors)

        for key in expected.keys():
            assert actual[key]
            for actual_arr, expected_arr in zip(actual[key], expected[key]):
                assert np.array_equal(actual_arr.sort(axis=0), expected_arr.sort(axis=0))

    def test_to_polygon_annotation_with_colors_in_meta(self):
        annotation = make_segmentation_representation(np.array(
            [[[128, 128, 128], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
             [[128, 128, 128], [128, 128, 128], [0, 0, 0], [0, 0, 0]],
             [[128, 128, 128], [128, 128, 128], [128, 128, 128], [0, 0, 0]]]), True)[0]
        dataset_meta = {'segmentation_colors': [[0, 0, 0], [128, 128, 128]]}
        annotation.metadata.update({
            'dataset_meta': dataset_meta
        })
        expected = {
            0: [np.array([[1, 0], [3, 0], [3, 2]])],
            1: [np.array([[0, 0], [0, 2], [2, 2]])]}

        actual = annotation.to_polygon()

        for key in expected.keys():
            assert actual[key]
            for actual_arr, expected_arr in zip(actual[key], expected[key]):
                assert np.array_equal(actual_arr.sort(axis=0), expected_arr.sort(axis=0))

    def test_to_polygon_annotation_with_colors_on_converted_annotation(self):
        annotation = make_segmentation_representation(np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]), True)[0]
        dataset_meta = {'segmentation_colors': [[0, 0, 0], [128, 128, 128]]}
        annotation.metadata.update({
            'dataset_meta': dataset_meta
        })
        expected = {
            0: [np.array([[1, 0], [3, 0], [3, 2]])],
            1: [np.array([[0, 0], [0, 2], [2, 2]])]}

        actual = annotation.to_polygon()

        for key in expected.keys():
            assert actual[key]
            for actual_arr, expected_arr in zip(actual[key], expected[key]):
                assert np.array_equal(actual_arr.sort(axis=0), expected_arr.sort(axis=0))

    def test_to_polygon_annotation_without_colors(self):
        annotation = make_segmentation_representation(np.array(
            [[[128, 128, 128], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
             [[128, 128, 128], [128, 128, 128], [0, 0, 0], [0, 0, 0]],
             [[128, 128, 128], [128, 128, 128], [128, 128, 128], [0, 0, 0]]]), True)[0]

        with pytest.raises(ValueError):
            annotation.to_polygon()

    def test_to_polygon_annotation_with_empty_mask(self):
        annotation = make_segmentation_representation(np.array([]), True)[0]

        with pytest.warns(Warning):
            assert len(annotation.to_polygon()) == 0

    def test_to_polygon_annotation_with_label_map_containing_not_all_classes(self):
        annotation = make_segmentation_representation(np.array(
            [[1, 0, 0, 0, 2], [1, 1, 0, 0, 2], [1, 1, 1, 0, 2]]), True)[0]
        dataset_meta = {'label_map': {0: "background", 1: "triangle"}}
        annotation.metadata.update({
            'dataset_meta': dataset_meta
        })

        expected = {
            0: [np.array([[1, 0], [3, 0], [3, 2]])],
            1: [np.array([[0, 0], [0, 2], [2, 2]])]}

        actual = annotation.to_polygon()

        for key in expected.keys():
            assert actual[key]
            for actual_arr, expected_arr in zip(actual[key], expected[key]):
                assert np.array_equal(actual_arr.sort(axis=0), expected_arr.sort(axis=0))
        assert actual.get(2) is None


    def test_to_polygon_prediction(self):
        prediction = make_segmentation_representation(np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]), False)[0]
        expected = {
            0: [np.array([[1, 0], [3, 0], [3, 2]])],
            1: [np.array([[0, 0], [0, 2], [2, 2]])]}

        actual = prediction.to_polygon()

        for key in expected.keys():
            assert actual[key]
            for actual_arr, expected_arr in zip(actual[key], expected[key]):
                assert np.array_equal(actual_arr.sort(axis=0), expected_arr.sort(axis=0))

    def test_to_polygon_prediction_with_argmax(self):
        prediction = make_segmentation_representation(np.array(
            [[[0.01, 0.99, 0.99, 0.99], [0.01, 0.01, 0.99, 0.99], [0.01, 0.01, 0.01, 0.99]],
             [[0.99, 0.01, 0.01, 0.01], [0.99, 0.99, 0.01, 0.01], [0.99, 0.99, 0.99, 0.01]]]), False)[0]
        expected = {
            0: [np.array([[1, 0], [3, 0], [3, 2]])],
            1: [np.array([[0, 0], [0, 2], [2, 2]])]}

        actual = prediction.to_polygon()

        for key in expected.keys():
            assert actual[key]
            for actual_arr, expected_arr in zip(actual[key], expected[key]):
                assert np.array_equal(actual_arr.sort(axis=0), expected_arr.sort(axis=0))

    def test_to_polygon_prediction_with_1_in_shape_channels_last(self):
        prediction = make_segmentation_representation(np.array(
            [[[1], [0], [0], [0]], [[1], [1], [0], [0]], [[1], [1], [1], [0]]]), False)[0]
        expected = {
            0: [np.array([[1, 0], [3, 0], [3, 2]])],
            1: [np.array([[0, 0], [0, 2], [2, 2]])]}

        actual = prediction.to_polygon()

        for key in expected.keys():
            assert actual[key]
            for actual_arr, expected_arr in zip(actual[key], expected[key]):
                assert np.array_equal(actual_arr.sort(axis=0), expected_arr.sort(axis=0))

    def test_to_polygon_prediction_with_1_in_shape_channels_first(self):
        prediction = make_segmentation_representation(np.array(
            [[[1], [0], [0], [0]], [[1], [1], [0], [0]], [[1], [1], [1], [0]]]).reshape(1, 3, 4), False)[0]
        expected = {
            0: [np.array([[1, 0], [3, 0], [3, 2]])],
            1: [np.array([[0, 0], [0, 2], [2, 2]])]}

        actual = prediction.to_polygon()

        for key in expected.keys():
            assert actual[key]
            for actual_arr, expected_arr in zip(actual[key], expected[key]):
                assert np.array_equal(actual_arr.sort(axis=0), expected_arr.sort(axis=0))

    def test_to_polygon_prediction_with_None_mask(self):
        prediction = make_segmentation_representation(None, False)[0]

        with pytest.warns(Warning):
            assert len(prediction.to_polygon()) == 0

    def test_to_polygon_prediction_with_empty_mask(self):
        prediction = make_segmentation_representation(np.array([]), False)[0]

        with pytest.warns(Warning):
            assert len(prediction.to_polygon()) == 0


@pytest.mark.skipif(no_available_pycocotools(), reason='no installed pycocotools in the system')
class TestCoCoInstanceSegmentationRepresentation:
    def test_to_polygon_annotation_mask_rle(self):
        mask = [np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]),
                np.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]])]
        raw_mask = encode_mask(mask)
        labels = [0, 1]
        annotation = make_instance_segmentation_representation(raw_mask, labels, True)[0]
        expected = {
            1: [np.array([[[1, 0], [3, 0], [3, 2]]])],
            0: [np.array([[[0, 0], [0, 2], [2, 2]]])]}

        actual = annotation.to_polygon()

        for key in expected.keys():
            assert actual[key]
            for actual_arr, expected_arr in zip(actual[key], expected[key]):
                actual_arr = np.sort(actual_arr, axis=1)
                expected_arr = np.sort(expected_arr, axis=1)
                assert np.array_equal(actual_arr, expected_arr)

    def test_to_polygon_annotation_mask_polygon(self):
        mask = [np.array([[[0, 0], [0, 2], [2, 2]]]),
                np.array([[[1, 0], [3, 0], [3, 2]]])]
        labels = [0, 1]
        annotation = make_instance_segmentation_representation(mask, labels, True)[0]
        expected = {
            1: [np.array([[[1, 0], [3, 0], [3, 2]]])],
            0: [np.array([[[0, 0], [0, 2], [2, 2]]])]}

        actual = annotation.to_polygon()

        for key in expected.keys():
            assert actual[key]
            for actual_arr, expected_arr in zip(actual[key], expected[key]):
                actual_arr = np.sort(actual_arr, axis=1)
                expected_arr = np.sort(expected_arr, axis=1)
                assert np.array_equal(actual_arr, expected_arr)

    def test_to_polygon_prediction(self):
        mask = [np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]),
                np.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]])]
        raw_mask = encode_mask(mask)
        labels = [0, 1]
        prediction = make_instance_segmentation_representation(raw_mask, labels, False)[0]
        expected = {
            1: [np.array([[[1, 0], [3, 0], [3, 2]]])],
            0: [np.array([[[0, 0], [0, 2], [2, 2]]])]}

        actual = prediction.to_polygon()

        for key in expected.keys():
            assert actual[key]
            for actual_arr, expected_arr in zip(actual[key], expected[key]):
                actual_arr = np.sort(actual_arr, axis=1)
                expected_arr = np.sort(expected_arr, axis=1)
                assert np.array_equal(actual_arr, expected_arr)

    def test_to_polygon_with_None_mask(self):
        labels = [0, 1]
        prediction = make_instance_segmentation_representation(None, labels, False)[0]

        with pytest.warns(Warning):
            assert len(prediction.to_polygon()) == 0

    def test_to_polygon_with_empty_mask(self):
        labels = [0, 1]
        prediction = make_instance_segmentation_representation([], labels, False)[0]

        with pytest.warns(Warning):
            assert len(prediction.to_polygon()) == 0

    def test_to_polygon_with_None_labels(self):
        prediction = make_instance_segmentation_representation([np.array([[1, 0], [0, 1]])], None, False)[0]

        with pytest.warns(Warning):
            assert len(prediction.to_polygon()) == 0

    def test_to_polygon_with_empty_labels(self):
        prediction = make_instance_segmentation_representation([np.array([[1, 0], [0, 1]])], [], False)[0]

        with pytest.warns(Warning):
            assert len(prediction.to_polygon()) == 0
