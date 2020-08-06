"""
Copyright (c) 2018-2020 Intel Corporation

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

from .common import make_segmentation_representation, make_instance_segmentation_representation


class TestSegmentationRepresentation:

    def test_to_polygon_annotation(self):
        annotations = make_segmentation_representation(np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]), True)
        expected = {
            0: [np.array([[1, 0], [3, 0], [3, 2]], dtype=np.int32)],
            1: [np.array([[0, 0], [0, 2], [2, 2]], dtype=np.int32)]}
        for annotation in annotations:
            actual = annotation.to_polygon()

        for key, _ in expected.items():
            assert actual[key]
            for actual_arr, expected_arr in zip(actual[key], expected[key]):
                assert np.array_equal(actual_arr.sort(axis=0), expected_arr.sort(axis=0))

    def test_to_polygon_annotation_with_colors(self):
        annotations = make_segmentation_representation(np.array(
            [[[128, 128, 128], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
             [[128, 128, 128], [128, 128, 128], [0, 0, 0], [0, 0, 0]],
             [[128, 128, 128], [128, 128, 128], [128, 128, 128], [0, 0, 0]]]), True)
        segmentation_colors = [[0, 0, 0], [128, 128, 128]]
        expected = {
            0: [np.array([[1, 0], [3, 0], [3, 2]], dtype=np.int32)],
            1: [np.array([[0, 0], [0, 2], [2, 2]], dtype=np.int32)]}
        for annotation in annotations:
            actual = annotation.to_polygon(segmentation_colors)

        for key, _ in expected.items():
            assert actual[key]
            for actual_arr, expected_arr in zip(actual[key], expected[key]):
                assert np.array_equal(actual_arr.sort(axis=0), expected_arr.sort(axis=0))

    def test_to_polygon_prediction(self):
        predictions = make_segmentation_representation(np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]), False)
        expected = {
            0: [np.array([[1, 0], [3, 0], [3, 2]], dtype=np.int32)],
            1: [np.array([[0, 0], [0, 2], [2, 2]], dtype=np.int32)]}
        for prediction in predictions:
            actual = prediction.to_polygon()

        for key, _ in expected.items():
            assert actual[key]
            for actual_arr, expected_arr in zip(actual[key], expected[key]):
                assert np.array_equal(actual_arr.sort(axis=0), expected_arr.sort(axis=0))

    def test_to_polygon_prediction_with_argmax(self):
        predictions = make_segmentation_representation(np.array(
            [[[0.01, 0.99, 0.99, 0.99], [0.01, 0.01, 0.99, 0.99], [0.01, 0.01, 0.01, 0.99]],
             [[0.99, 0.01, 0.01, 0.01], [0.99, 0.99, 0.01, 0.01], [0.99, 0.99, 0.99, 0.01]]]), False)
        expected = {
            0: [np.array([[1, 0], [3, 0], [3, 2]], dtype=np.int32)],
            1: [np.array([[0, 0], [0, 2], [2, 2]], dtype=np.int32)]}
        for prediction in predictions:
            actual = prediction.to_polygon()

        for key, _ in expected.items():
            assert actual[key]
            for actual_arr, expected_arr in zip(actual[key], expected[key]):
                assert np.array_equal(actual_arr.sort(axis=0), expected_arr.sort(axis=0))

class TestCoCoInstanceSegmentationRepresentation:

    def test_to_polygon_annotation(self):
        mask = [np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]),
                np.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]])]
        labels = [0, 1]
        annotations = make_instance_segmentation_representation(mask, labels, True)
        expected = {
            0: [np.array([[1, 0], [3, 0], [3, 2]], dtype=np.int32)],
            1: [np.array([[0, 0], [0, 2], [2, 2]], dtype=np.int32)]}
        for annotation in annotations:
            actual = annotation.to_polygon()

        for key, _ in expected.items():
            assert actual[key]
            for actual_arr, expected_arr in zip(actual[key], expected[key]):
                assert np.array_equal(actual_arr.sort(axis=0), expected_arr.sort(axis=0))

    def test_to_polygon_prediction(self):
        mask = [np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]),
                np.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]])]
        labels = [0, 1]
        predictions = make_instance_segmentation_representation(mask, labels, False)
        expected = {
            0: [np.array([[1, 0], [3, 0], [3, 2]], dtype=np.int32)],
            1: [np.array([[0, 0], [0, 2], [2, 2]], dtype=np.int32)]}
        for prediction in predictions:
            actual = prediction.to_polygon()

        for key, _ in expected.items():
            assert actual[key]
            for actual_arr, expected_arr in zip(actual[key], expected[key]):
                assert np.array_equal(actual_arr.sort(axis=0), expected_arr.sort(axis=0))
