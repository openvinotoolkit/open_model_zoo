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

import numpy as np
from PIL import Image

from accuracy_checker.evaluators.model_evaluator import ModelEvaluator
from accuracy_checker.representation.image_processing import GTLoader, ImageProcessingAnnotation


def make_annotation(tmp_path):
    expected = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    Image.fromarray(expected).save(tmp_path / 'gt.png')
    annotation = ImageProcessingAnnotation('gt.png', 'gt.png', GTLoader.PILLOW)
    annotation.metadata = {'data_source': tmp_path}
    return annotation, expected


def test_release_cached_value_drops_lazily_loaded_image(tmp_path):
    annotation, expected = make_annotation(tmp_path)

    assert np.array_equal(annotation.value, expected)
    assert annotation._value is not None  # pylint: disable=protected-access

    annotation.release_cached_value()
    assert annotation._value is None  # pylint: disable=protected-access
    assert np.array_equal(annotation.value, expected)


def test_release_cached_value_keeps_explicitly_set_value(tmp_path):
    annotation, _ = make_annotation(tmp_path)
    processed = np.zeros((2, 2, 3), dtype=np.uint8)
    annotation.value = processed

    annotation.release_cached_value()
    assert annotation.value is processed


def test_model_evaluator_releases_annotations_and_ignores_others(tmp_path):
    annotation, expected = make_annotation(tmp_path)
    assert np.array_equal(annotation.value, expected)

    ModelEvaluator._release_cached_annotations([annotation, None, object()], None)  # pylint: disable=protected-access
    assert annotation._value is None  # pylint: disable=protected-access
