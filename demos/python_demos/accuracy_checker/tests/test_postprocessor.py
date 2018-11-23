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
import pytest

from accuracy_checker.config import ConfigError
from accuracy_checker.postprocessor import PostprocessingExecutor
from accuracy_checker.representation import (DetectionAnnotation, DetectionPrediction,
                                             ContainerAnnotation, ContainerPrediction, ClassificationAnnotation)


def postprocess_data(executor, annotations, predictions):
    return executor.full_process(annotations, predictions)


class TestPostprocessor:
    def test_without_apply_to_and_sources_filter_raise_config_error_exception(self):
        config = [{'type': 'filter', 'labels': ['to_be_filtered']}]

        with pytest.raises(ConfigError):
            PostprocessingExecutor(config)

    def test_both_provided_apply_to_and_sources_filter_raise_config_error_exception(self):
        config = [{'type': 'filter', 'apply_to': 'prediction',
                   'annotation_source': 'annotation', 'labels': ['to_be_filtered']}]

        with pytest.raises(ConfigError):
            PostprocessingExecutor(config)

    def test_filter_annotations_unsupported_source_type_in_container_raise_type_error_exception(self):
        config = [{'type': 'filter', 'annotation_source': 'annotation', 'labels': ['to_be_filtered']}]
        annotation = ContainerAnnotation({'annotation': ClassificationAnnotation()})
        executor = PostprocessingExecutor(config)

        with pytest.raises(TypeError):
            postprocess_data(executor, [annotation], [None])

    def test_filter_annotations_source_not_found_raise_config_error_exception(self):
        config = [{'type': 'filter', 'annotation_source': 'ann', 'labels': ['to_be_filtered']}]
        annotation = ContainerAnnotation({'annotation': DetectionAnnotation(labels=['some_label', 'to_be_filtered'])})
        executor = PostprocessingExecutor(config)

        with pytest.raises(ConfigError):
            postprocess_data(executor, [annotation], [None])

    def test_filter_predictions_unsupported_source_type_raise_type_error_exception(self):
        config = [{'type': 'filter', 'prediction_source': 'detection_out', 'labels': ['to_be_filtered'],
                   'remove_filtered': False}]
        prediction = ContainerPrediction({'detection_out': ClassificationAnnotation()})
        executor = PostprocessingExecutor(config)

        with pytest.raises(TypeError):
            postprocess_data(executor, [None], [prediction])

    def test_filter_predictions_source_not_found_raise_config_error_exception(self):
        config = [{'type': 'filter', 'prediction_source': 'undefined', 'labels': ['to_be_filtered']}]
        prediction = ContainerPrediction(
            {'detection_out': DetectionPrediction(labels=['some_label', 'to_be_filtered'])}
        )
        executor = PostprocessingExecutor(config)

        with pytest.raises(ConfigError):
            postprocess_data(executor, [None], [prediction])

    def test_filter_container_annotations_by_labels_with_ignore_using_source(self):
        config = [{'type': 'filter', 'annotation_source': 'annotation',
                   'labels': ['to_be_filtered'], 'remove_filtered': False}]
        annotation = ContainerAnnotation({'annotation': DetectionAnnotation(labels=['some_label', 'to_be_filtered'])})
        expected = ContainerAnnotation({'annotation': DetectionAnnotation(labels=['some_label', 'to_be_filtered'],
                                                                          metadata={'difficult_boxes': [1]})})

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_filter_container_annotations_by_labels_with_ignore_using_apply_to(self):
        config = [{'type': 'filter', 'apply_to': 'annotation', 'labels': ['to_be_filtered'],
                   'remove_filtered': False}]
        annotation = ContainerAnnotation(
            {'annotation': DetectionAnnotation(labels=['some_label', 'to_be_filtered'])}
        )
        expected = ContainerAnnotation({'annotation': DetectionAnnotation(labels=['some_label', 'to_be_filtered'],
                                                                          metadata={'difficult_boxes': [1]})})

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_filter_regular_annotations_by_labels_with_ignore(self):
        config = [{'type': 'filter', 'apply_to': 'annotation', 'labels': ['to_be_filtered'], 'remove_filtered': False}]
        annotation = DetectionAnnotation(labels=['some_label', 'to_be_filtered'])
        expected = DetectionAnnotation(labels=['some_label', 'to_be_filtered'], metadata={'difficult_boxes': [1]})

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_filter_multi_source_annotations_by_labels_with_ignore(self):
        config = [{'type': 'filter', 'annotation_source': ['annotation1', 'annotation2'], 'labels': ['to_be_filtered'],
                   'remove_filtered': False}]
        annotation = ContainerAnnotation({'annotation1': DetectionAnnotation(labels=['some_label', 'to_be_filtered']),
                                          'annotation2': DetectionAnnotation(labels=['some_label', 'to_be_filtered'])})
        expected = ContainerAnnotation(
            {
                'annotation1': DetectionAnnotation(
                    labels=['some_label', 'to_be_filtered'], metadata={'difficult_boxes': [1]}
                ),
                'annotation2': DetectionAnnotation(
                    labels=['some_label', 'to_be_filtered'], metadata={'difficult_boxes': [1]}
                )
            }
        )

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_filter_multi_source_annotations_by_labels_with_ignore_using_apply_to(self):
        config = [{'type': 'filter', 'apply_to': 'annotation',  'labels': ['to_be_filtered'],
                   'remove_filtered': False}]
        annotation = ContainerAnnotation({'annotation1': DetectionAnnotation(labels=['some_label', 'to_be_filtered']),
                                          'annotation2': DetectionAnnotation(labels=['some_label', 'to_be_filtered'])})
        expected = ContainerAnnotation(
            {
                'annotation1': DetectionAnnotation(
                    labels=['some_label', 'to_be_filtered'], metadata={'difficult_boxes': [1]}
                ),
                'annotation2': DetectionAnnotation(
                    labels=['some_label', 'to_be_filtered'], metadata={'difficult_boxes': [1]}
                )
            }
        )
        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_filter_regular_annotations_by_labels_with_remove_using_annotation_source_warm_user_warning(self):
        config = [{'type': 'filter', 'annotation_source': 'annotation',
                   'labels': ['to_be_filtered'], 'remove_filtered': True}]
        annotation = DetectionAnnotation(labels=['some_label', 'to_be_filtered'])
        expected = DetectionAnnotation(labels=['some_label'])

        with pytest.warns(UserWarning):
            postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_filter_regular_annotations_by_labels_with_remove_using_apply_to(self):
        config = [{'type': 'filter', 'apply_to': 'annotation', 'labels': ['to_be_filtered'], 'remove_filtered': True}]
        annotation = DetectionAnnotation(labels=['some_label', 'to_be_filtered'])
        expected = DetectionAnnotation(labels=['some_label'])

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_filter_annotations_by_labels_with_remove_on_container(self):
        config = [{'type': 'filter', 'annotation_source': 'annotation',
                   'labels': ['to_be_filtered'], 'remove_filtered': True}]
        annotation = ContainerAnnotation({'annotation': DetectionAnnotation(labels=['some_label', 'to_be_filtered'])})
        expected = ContainerAnnotation({'annotation': DetectionAnnotation(labels=['some_label'])})

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_filter_annotations_by_labels_with_remove_on_container_using_apply_to(self):
        config = [{'type': 'filter', 'apply_to': 'annotation', 'labels': ['to_be_filtered'], 'remove_filtered': True}]
        annotation = ContainerAnnotation({'annotation': DetectionAnnotation(labels=['some_label', 'to_be_filtered'])})
        expected = ContainerAnnotation({'annotation': DetectionAnnotation(labels=['some_label'])})

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_filter_multi_source_annotations_by_labels_with_remove(self):
        config = [{'type': 'filter', 'annotation_source': ['annotation1', 'annotation2'],
                   'labels': ['to_be_filtered'], 'remove_filtered': True}]
        annotation = ContainerAnnotation({'annotation1': DetectionAnnotation(labels=['some_label', 'to_be_filtered']),
                                          'annotation2': DetectionAnnotation(labels=['some_label', 'to_be_filtered'])})
        expected = ContainerAnnotation({'annotation1': DetectionAnnotation(labels=['some_label']),
                                        'annotation2': DetectionAnnotation(labels=['some_label'])})

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_filter_multi_source_by_labels_with_remove_on_container_using_apply_to(self):
        config = [{'type': 'filter', 'apply_to': 'annotation', 'labels': ['to_be_filtered'], 'remove_filtered': True}]
        annotation = ContainerAnnotation({'annotation1': DetectionAnnotation(labels=['some_label', 'to_be_filtered']),
                                          'annotation2': DetectionAnnotation(labels=['some_label', 'to_be_filtered'])})
        expected = ContainerAnnotation({'annotation1': DetectionAnnotation(labels=['some_label']),
                                        'annotation2': DetectionAnnotation(labels=['some_label'])})

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_filter_predictions_by_labels_with_ignore(self):
        config = [{'type': 'filter', 'apply_to': 'prediction', 'labels': ['to_be_filtered'], 'remove_filtered': False}]
        prediction = DetectionPrediction(labels=['some_label', 'to_be_filtered'])
        expected = DetectionPrediction(labels=['some_label', 'to_be_filtered'], metadata={'difficult_boxes': [1]})

        postprocess_data(PostprocessingExecutor(config), [None], [prediction])

        assert prediction == expected

    def test_filter_predictions_by_labels_with_ignore_on_container(self):
        config = [{'type': 'filter', 'prediction_source': 'detection_out',
                   'labels': ['to_be_filtered'], 'remove_filtered': False}]
        prediction = ContainerPrediction(
            {'detection_out': DetectionPrediction(labels=['some_label', 'to_be_filtered'])}
        )
        expected = ContainerPrediction(
            {'detection_out': DetectionPrediction(labels=['some_label', 'to_be_filtered'],
                                                  metadata={'difficult_boxes': [1]})}
        )

        postprocess_data(PostprocessingExecutor(config), [None], [prediction])

        assert prediction == expected

    def test_filter_predictions_by_labels_with_ignore_on_container_using_apply_to(self):
        config = [{'type': 'filter', 'apply_to': 'prediction', 'labels': ['to_be_filtered'], 'remove_filtered': False}]
        prediction = ContainerPrediction(
            {'detection_out': DetectionPrediction(labels=['some_label', 'to_be_filtered'])}
        )
        expected = ContainerPrediction(
            {'detection_out': DetectionPrediction(labels=['some_label', 'to_be_filtered'],
                                                  metadata={'difficult_boxes': [1]})}
        )

        postprocess_data(PostprocessingExecutor(config), [None], [prediction])

        assert prediction == expected

    def test_filter_multi_source_predictions_by_labels_with_ignore(self):
        config = [
            {'type': 'filter', 'prediction_source': ['detection_out1', 'detection_out2'], 'labels': ['to_be_filtered'],
             'remove_filtered': False}]
        prediction = ContainerPrediction(
            {'detection_out1': DetectionPrediction(labels=['some_label', 'to_be_filtered']),
             'detection_out2': DetectionPrediction(labels=['some_label', 'to_be_filtered'])})
        expected = ContainerPrediction(
            {
                'detection_out1': DetectionPrediction(labels=['some_label', 'to_be_filtered'],
                                                      metadata={'difficult_boxes': [1]}),
                'detection_out2': DetectionPrediction(labels=['some_label', 'to_be_filtered'],
                                                      metadata={'difficult_boxes': [1]})
            }
        )

        postprocess_data(PostprocessingExecutor(config), [None], [prediction])

        assert prediction == expected

    def test_filter_multi_source_predictions_by_labels_with_ignore_using_apply_to(self):
        config = [
            {'type': 'filter', 'apply_to': 'prediction', 'labels': ['to_be_filtered'],
             'remove_filtered': False}]
        prediction = ContainerPrediction(
            {'detection_out1': DetectionPrediction(labels=['some_label', 'to_be_filtered']),
             'detection_out2': DetectionPrediction(labels=['some_label', 'to_be_filtered'])})
        expected = ContainerPrediction(
            {
                'detection_out1': DetectionPrediction(labels=['some_label', 'to_be_filtered'],
                                                      metadata={'difficult_boxes': [1]}),
                'detection_out2': DetectionPrediction(labels=['some_label', 'to_be_filtered'],
                                                      metadata={'difficult_boxes': [1]})
            }
        )

        postprocess_data(PostprocessingExecutor(config), [None], [prediction])

        assert prediction == expected

    def test_filter_predictions_by_labels_with_remove(self):
        config = [{'type': 'filter', 'apply_to': 'prediction', 'labels': ['to_be_filtered'], 'remove_filtered': True}]
        prediction = DetectionPrediction(labels=['some_label', 'to_be_filtered'])
        expected = DetectionPrediction(labels=['some_label'])

        postprocess_data(PostprocessingExecutor(config), [None], [prediction])

        assert prediction == expected

    def test_filter_predictions_by_labels_with_remove_on_container(self):
        config = [{'type': 'filter', 'prediction_source': 'detection_out',
                   'labels': ['to_be_filtered'], 'remove_filtered': True}]
        prediction = ContainerPrediction(
            {'detection_out': DetectionPrediction(labels=['some_label', 'to_be_filtered'])}
        )
        expected = ContainerPrediction({'detection_out': DetectionPrediction(labels=['some_label'])})

        postprocess_data(PostprocessingExecutor(config), [None], [prediction])

        assert prediction == expected

    def test_filter_predictions_by_labels_with_remove_on_container_using_apply_to(self):
        config = [{'type': 'filter', 'apply_to': 'prediction', 'labels': ['to_be_filtered'], 'remove_filtered': True}]
        prediction = ContainerPrediction(
            {'detection_out': DetectionPrediction(labels=['some_label', 'to_be_filtered'])}
        )
        expected = ContainerPrediction({'detection_out': DetectionPrediction(labels=['some_label'])})

        postprocess_data(PostprocessingExecutor(config), [None], [prediction])

        assert prediction == expected

    def test_filter_multi_source_predictions_by_labels_with_remove(self):
        config = [{'type': 'filter', 'prediction_source': ['detection_out1', 'detection_out2'],
                   'labels': ['to_be_filtered'], 'remove_filtered': True}]
        prediction = ContainerPrediction(
            {'detection_out1': DetectionPrediction(labels=['some_label', 'to_be_filtered']),
             'detection_out2': DetectionPrediction(labels=['some_label', 'to_be_filtered'])}
        )
        expected = ContainerPrediction(
            {'detection_out1': DetectionPrediction(labels=['some_label']),
             'detection_out2': DetectionPrediction(labels=['some_label'])}
        )

        postprocess_data(PostprocessingExecutor(config), [None], [prediction])

        assert prediction == expected

    def test_filter_multi_source_predictions_by_labels_with_remove_using_apply_to(self):
        config = [{'type': 'filter', 'apply_to': 'prediction', 'labels': ['to_be_filtered'], 'remove_filtered': True}]
        prediction = ContainerPrediction(
            {'detection_out1': DetectionPrediction(labels=['some_label', 'to_be_filtered']),
             'detection_out2': DetectionPrediction(labels=['some_label', 'to_be_filtered'])}
        )
        expected = ContainerPrediction(
            {'detection_out1': DetectionPrediction(labels=['some_label']),
             'detection_out2': DetectionPrediction(labels=['some_label'])}
        )

        postprocess_data(PostprocessingExecutor(config), [None], [prediction])

        assert prediction == expected

    def test_filter_regular_annotations_and_regular_predictions_by_labels_with_ignore_using_apply_to(self):
        config = [{'type': 'filter', 'apply_to': 'all', 'labels': ['to_be_filtered'], 'remove_filtered': False}]
        prediction = DetectionPrediction(labels=['some_label', 'to_be_filtered'])
        expected_prediction = DetectionPrediction(labels=['some_label', 'to_be_filtered'],
                                                  metadata={'difficult_boxes': [1]})
        annotation = DetectionAnnotation(labels=['some_label', 'to_be_filtered'])
        expected_annotation = DetectionAnnotation(labels=['some_label', 'to_be_filtered'],
                                                  metadata={'difficult_boxes': [1]})

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction and annotation == expected_annotation

    def test_filter_regular_annotations_and_regular_predictions_by_labels_with_remove_using_apply_to(self):
        config = [{'type': 'filter', 'apply_to': 'all', 'labels': ['to_be_filtered'], 'remove_filtered': True}]
        prediction = DetectionPrediction(labels=['some_label', 'to_be_filtered'])
        expected_prediction = DetectionPrediction(labels=['some_label'])
        annotation = DetectionAnnotation(labels=['some_label', 'to_be_filtered'])
        expected_annotation = DetectionAnnotation(labels=['some_label'])

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction and annotation == expected_annotation

    def test_filter_container_annotations_and_regular_predictions_by_labels_with_ignore_using_apply_to(self):
        config = [{'type': 'filter', 'apply_to': 'all', 'labels': ['to_be_filtered'], 'remove_filtered': False}]
        prediction = DetectionPrediction(labels=['some_label', 'to_be_filtered'])
        expected_prediction = DetectionPrediction(labels=['some_label', 'to_be_filtered'],
                                                  metadata={'difficult_boxes': [1]})
        annotation = ContainerAnnotation({'annotation': DetectionAnnotation(labels=['some_label', 'to_be_filtered'])})
        expected_annotation = ContainerAnnotation(
            {'annotation': DetectionAnnotation(labels=['some_label', 'to_be_filtered'],
                                               metadata={'difficult_boxes': [1]})}
        )

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction and annotation == expected_annotation

    def test_filter_container_annotations_and_regular_predictions_by_labels_with_remove_using_apply_to(self):
        config = [{'type': 'filter', 'apply_to': 'all', 'labels': ['to_be_filtered'], 'remove_filtered': True}]
        prediction = DetectionPrediction(labels=['some_label', 'to_be_filtered'])
        expected_prediction = DetectionPrediction(labels=['some_label'])
        annotation = ContainerAnnotation({'annotation': DetectionAnnotation(labels=['some_label', 'to_be_filtered'])})
        expected_annotation = ContainerAnnotation({'annotation': DetectionAnnotation(labels=['some_label'])})

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction and annotation == expected_annotation

    def test_filter_regular_annotations_and_container_predictions_by_labels_with_ignore_using_apply_to(self):
        config = [{'type': 'filter', 'apply_to': 'all', 'labels': ['to_be_filtered'], 'remove_filtered': False}]
        prediction = ContainerPrediction(
            {'detection_out': DetectionPrediction(labels=['some_label', 'to_be_filtered'])})
        expected_prediction = ContainerPrediction(
            {'detection_out': DetectionPrediction(labels=['some_label', 'to_be_filtered'],
                                                  metadata={'difficult_boxes': [1]})})
        annotation = DetectionAnnotation(labels=['some_label', 'to_be_filtered'])
        expected_annotation = DetectionAnnotation(labels=['some_label', 'to_be_filtered'],
                                                  metadata={'difficult_boxes': [1]})

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction and annotation == expected_annotation

    def test_filter_regular_annotations_and_container_predictions_by_labels_with_remove_using_apply_to(self):
        config = [{'type': 'filter', 'apply_to': 'all', 'labels': ['to_be_filtered'], 'remove_filtered': True}]
        prediction = ContainerPrediction(
            {'detection_out': DetectionPrediction(labels=['some_label', 'to_be_filtered'])})
        expected_prediction = ContainerPrediction({'detection_out': DetectionPrediction(labels=['some_label'])})
        annotation = DetectionAnnotation(labels=['some_label', 'to_be_filtered'])
        expected_annotation = DetectionAnnotation(labels=['some_label'])

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction and annotation == expected_annotation

    def test_filter_container_annotations_and_container_predictions_by_labels_with_ignore_using_apply_to(self):
        config = [{'type': 'filter', 'apply_to': 'all', 'labels': ['to_be_filtered'], 'remove_filtered': False}]
        prediction = ContainerPrediction(
            {'detection_out': DetectionPrediction(labels=['some_label', 'to_be_filtered'])})
        expected_prediction = ContainerPrediction(
            {'detection_out': DetectionPrediction(labels=['some_label', 'to_be_filtered'],
                                                  metadata={'difficult_boxes': [1]})}
        )
        annotation = ContainerAnnotation({'annotation': DetectionAnnotation(labels=['some_label', 'to_be_filtered'])})
        expected_annotation = ContainerAnnotation(
            {'annotation': DetectionAnnotation(labels=['some_label', 'to_be_filtered'],
                                               metadata={'difficult_boxes': [1]})}
        )

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction and annotation == expected_annotation

    def test_filter_container_annotations_and_container_predictions_by_labels_with_remove_using_apply_to(self):
        config = [{'type': 'filter', 'apply_to': 'all', 'labels': ['to_be_filtered'], 'remove_filtered': True}]
        prediction = ContainerPrediction({'prediction': DetectionPrediction(labels=['some_label', 'to_be_filtered'])})
        expected_prediction = ContainerPrediction({'prediction': DetectionPrediction(labels=['some_label'])})
        annotation = ContainerAnnotation({'annotation': DetectionAnnotation(labels=['some_label', 'to_be_filtered'])})
        expected_annotation = ContainerAnnotation({'annotation': DetectionAnnotation(labels=['some_label'])})

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction and annotation == expected_annotation

    def test_filter_container_annotations_and_container_predictions_by_labels_with_ignore_using_sources(self):
        config = [{'type': 'filter', 'apply_to': 'all', 'labels': ['to_be_filtered'], 'remove_filtered': False}]
        prediction = ContainerPrediction({'prediction': DetectionPrediction(labels=['some_label', 'to_be_filtered'])})
        expected_prediction = ContainerPrediction(
            {'prediction': DetectionPrediction(labels=['some_label', 'to_be_filtered'],
                                               metadata={'difficult_boxes': [1]})}
        )
        annotation = ContainerAnnotation({'annotation': DetectionAnnotation(labels=['some_label', 'to_be_filtered'])})
        expected_annotation = ContainerAnnotation(
            {'annotation': DetectionAnnotation(labels=['some_label', 'to_be_filtered'],
                                               metadata={'difficult_boxes': [1]})}
        )

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction and annotation == expected_annotation

    def test_filter_container_annotations_and_container_predictions_by_labels_with_remove_using_sources(self):
        config = [{'type': 'filter', 'annotation_source': 'annotation', 'prediction_source': 'prediction',
                   'labels': ['to_be_filtered'], 'remove_filtered': True}]
        prediction = ContainerPrediction({'prediction': DetectionPrediction(labels=['some_label', 'to_be_filtered'])})
        expected_prediction = ContainerPrediction({'prediction': DetectionPrediction(labels=['some_label'])})
        annotation = ContainerAnnotation({'annotation': DetectionAnnotation(labels=['some_label', 'to_be_filtered'])})
        expected_annotation = ContainerAnnotation({'annotation': DetectionAnnotation(labels=['some_label'])})

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction and annotation == expected_annotation

    def test_filter_annotations_by_min_confidence_do_nothing(self):
        config = [{'type': 'filter', 'apply_to': 'annotation', 'min_confidence': 0.5, 'remove_filtered': True}]
        annotations = [DetectionAnnotation(labels=['a', 'b']), DetectionAnnotation(labels=['c', 'd'])]
        expected_annotations = [DetectionAnnotation(labels=['a', 'b']), DetectionAnnotation(labels=['c', 'd'])]

        postprocess_data(PostprocessingExecutor(config), annotations, [None, None])

        assert np.array_equal(annotations, expected_annotations)

    def test_filter_predictions_by_min_confidence_with_ignore(self):
        config = [{'type': 'filter', 'apply_to': 'prediction', 'min_confidence': 0.5, 'remove_filtered': False}]
        predictions = [DetectionPrediction(scores=[0.3, 0.8]), DetectionPrediction(scores=[0.5, 0.4])]
        expected_predictions = [
            DetectionPrediction(scores=[0.3, 0.8], metadata={'difficult_boxes': [0]}),
            DetectionPrediction(scores=[0.5, 0.4], metadata={'difficult_boxes': [1]})
        ]

        executor = PostprocessingExecutor(config)
        postprocess_data(executor, [None, None], predictions)

        assert np.array_equal(predictions, expected_predictions)

    def test_filter_predictions_by_min_confidence_with_remove(self):
        config = [{'type': 'filter', 'apply_to': 'prediction', 'min_confidence': 0.5, 'remove_filtered': True}]
        predictions = [DetectionPrediction(scores=[0.3, 0.8]), DetectionPrediction(scores=[0.4, 0.5])]
        expected_predictions = [DetectionPrediction(scores=[0.8]), DetectionPrediction(scores=[0.5])]

        postprocess_data(PostprocessingExecutor(config), [None, None], predictions)

        assert np.array_equal(predictions, expected_predictions)

    def test_filter_annotations_by_height_range_with_ignored(self):
        config = [{'type': 'filter', 'apply_to': 'annotation', 'height_range': '(10.0, 20.0)',
                   'remove_filtered': False}]
        annotations = [
            DetectionAnnotation(y_mins=[5.0, 10.0], y_maxs=[15.0, 10.0]),
            DetectionAnnotation(y_mins=[5.0, 10.0], y_maxs=[35.0, 40.0])
        ]
        expected = [
            DetectionAnnotation(y_mins=[5.0, 10.0], y_maxs=[15.0, 10.0], metadata={'difficult_boxes': [1]}),
            DetectionAnnotation(y_mins=[5.0, 10.0], y_maxs=[35.0, 40.0], metadata={'difficult_boxes': [0, 1]})
        ]

        postprocess_data(PostprocessingExecutor(config), annotations, [None, None])

        assert np.array_equal(annotations, expected)

    def test_filter_annotations_by_height_range_with_remove(self):
        config = [{'type': 'filter', 'apply_to': 'annotation', 'height_range': '(10.0, 20.0)', 'remove_filtered': True}]
        annotations = [
            DetectionAnnotation(y_mins=[5.0, 10.0], y_maxs=[15.0, 10.0]),
            DetectionAnnotation(y_mins=[5.0, 10.0], y_maxs=[35.0, 40.0])
        ]
        expected = [
            DetectionAnnotation(y_mins=[5.0], y_maxs=[15.0]),
            DetectionAnnotation(y_mins=[], y_maxs=[])
        ]

        postprocess_data(PostprocessingExecutor(config), annotations, [None, None])

        assert np.array_equal(annotations, expected)

    def test_filter_predictions_by_height_range_with_ignored(self):
        config = [{'type': 'filter', 'apply_to': 'prediction',
                   'height_range': '(10.0, 20.0)', 'remove_filtered': False}]
        predictions = [
            DetectionPrediction(y_mins=[5.0, 10.0], y_maxs=[15.0, 40.0]),
            DetectionPrediction(y_mins=[5.0, 10.0], y_maxs=[35.0, 50.0])
        ]
        expected = [
            DetectionPrediction(y_mins=[5.0, 10.0], y_maxs=[15.0, 40.0], metadata={'difficult_boxes': [1]}),
            DetectionPrediction(y_mins=[5.0, 10.0], y_maxs=[35.0, 50.0], metadata={'difficult_boxes': [0, 1]})
        ]

        postprocess_data(PostprocessingExecutor(config), [None, None], predictions)

        assert np.array_equal(predictions, expected)

    def test_filter_predictions_by_height_range_with_remove(self):
        config = [{'type': 'filter', 'apply_to': 'prediction', 'height_range': '(10.0, 20.0)', 'remove_filtered': True}]
        predictions = [
            DetectionPrediction(y_mins=[5.0, 10.0], y_maxs=[15.0, 40.0]),
            DetectionPrediction(y_mins=[5.0, 10.0], y_maxs=[35.0, 50.0])
        ]
        expected = [
            DetectionPrediction(y_mins=[5.0], y_maxs=[15.0]),
            DetectionPrediction(y_mins=[], y_maxs=[])
        ]

        postprocess_data(PostprocessingExecutor(config), [None, None], predictions)

        assert np.array_equal(predictions, expected)

    def test_filter_by_unknown_visibility_does_nothing_with_predictions(self):
        config = [{'type': 'filter', 'apply_to': 'prediction', 'min_visibility': 'unknown'}]
        predictions = [
           DetectionPrediction(y_mins=[5.0, 10.0], y_maxs=[15.0, 40.0]),
           DetectionPrediction(y_mins=[5.0, 10.0], y_maxs=[35.0, 50.0])
        ]
        expected = [
           DetectionPrediction(y_mins=[5.0, 10.0], y_maxs=[15.0, 40.0], metadata={'difficult_boxes': []}),
           DetectionPrediction(y_mins=[5.0, 10.0], y_maxs=[35.0, 50.0], metadata={'difficult_boxes': []})
        ]

        postprocess_data(PostprocessingExecutor(config), [None, None], predictions)

        assert np.array_equal(predictions, expected)

    def test_filter_by_visibility_does_nothing_with_annotations_without_visibility(self):
        config = [{'type': 'filter', 'apply_to': 'annotation', 'min_visibility': 'heavy occluded'}]
        annotations = [
            DetectionAnnotation(y_mins=[5.0, 10.0], y_maxs=[15.0, 40.0]),
            DetectionAnnotation(y_mins=[5.0, 10.0], y_maxs=[35.0, 50.0])
        ]
        expected = [
            DetectionAnnotation(y_mins=[5.0, 10.0], y_maxs=[15.0, 40.0], metadata={'difficult_boxes': []}),
            DetectionAnnotation(y_mins=[5.0, 10.0], y_maxs=[35.0, 50.0], metadata={'difficult_boxes': []})
        ]

        postprocess_data(PostprocessingExecutor(config), annotations, [None, None])

        assert np.array_equal(annotations, expected)

    def test_filter_by_visibility_does_nothing_with_default_visibility_level_and_heavy_occluded(self):
        config = [{'type': 'filter', 'apply_to': 'annotation', 'min_visibility': 'heavy occluded'}]
        annotation = DetectionAnnotation(y_mins=[5.0, 10.0], y_maxs=[15.0, 40.0])
        expected = DetectionAnnotation(y_mins=[5.0, 10.0], y_maxs=[15.0, 40.0], metadata={'difficult_boxes': []})

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_filter_by_visibility_does_nothing_with_default_visibility_level_and_partially_occluded(self):
        config = [{'type': 'filter', 'apply_to': 'annotation', 'min_visibility': 'partially occluded'}]
        annotation = DetectionAnnotation(y_mins=[5.0, 10.0], y_maxs=[15.0, 40.0])
        expected = DetectionAnnotation(y_mins=[5.0, 10.0], y_maxs=[15.0, 40.0], metadata={'difficult_boxes': []})

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_filter_by_visibility_filters_partially_occluded(self):
        config = [{'type': 'filter', 'apply_to': 'annotation', 'min_visibility': 'partially occluded',
                   'remove_filtered': True}]
        annotation = DetectionAnnotation(
            y_mins=[5.0, 10.0], y_maxs=[15.0, 40.0], metadata={'visibilities': ['heavy occluded', 'partially occluded']}
        )
        expected = DetectionAnnotation(
            y_mins=[10.0], y_maxs=[40.0], metadata={'visibilities': ['heavy occluded', 'partially occluded']}
        )

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_nms(self, mocker):
        mock = mocker.patch('accuracy_checker.postprocessor.nms.NMS.process_all', return_value=([], []))
        config = [{'type': 'nms', 'overlap': 0.4}]
        postprocessing_evaluator = PostprocessingExecutor(config)

        postprocess_data(postprocessing_evaluator, [], [])

        assert len(postprocessing_evaluator._processors) == 1
        nms = postprocessing_evaluator._processors[0]
        assert nms.overlap == .4
        mock.assert_called_once_with([], [])

    def test_resize_prediction_boxes(self):
        config = [{'type': 'resize_prediction_boxes'}]
        annotation = DetectionAnnotation(metadata={'image_size': (100, 100, 3)})
        prediction = DetectionPrediction(x_mins=[0, 7], y_mins=[0, 7], x_maxs=[5, 8], y_maxs=[5, 8])
        expected_prediction = DetectionPrediction(
            x_mins=[pytest.approx(0), pytest.approx(700)],
            y_mins=[pytest.approx(0), pytest.approx(700)],
            x_maxs=[pytest.approx(500), pytest.approx(800)],
            y_maxs=[pytest.approx(500), pytest.approx(800)]
        )

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction

    def test_clip_annotation_denormalized_boxes(self):
        config = [{'type': 'clip_boxes', 'apply_to': 'annotation', 'boxes_normalized': False}]
        meta = {'image_size': (10, 10, 3)}
        annotation = DetectionAnnotation(x_mins=[-1, 9], y_mins=[0, 11], x_maxs=[5, 10], y_maxs=[5, 10], metadata=meta)
        expected = DetectionAnnotation(
            x_mins=[pytest.approx(0), pytest.approx(9)],
            y_mins=[pytest.approx(0), pytest.approx(10)],
            x_maxs=[pytest.approx(5), pytest.approx(10)],
            y_maxs=[pytest.approx(5), pytest.approx(10)],
            metadata=meta
        )

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_clip_annotation_normalized_boxes(self):
        config = [{'type': 'clip_boxes', 'apply_to': 'annotation', 'boxes_normalized': True}]
        meta = {'image_size': (10, 10, 3)}
        annotation = DetectionAnnotation(x_mins=[-1, 9], y_mins=[0, 11], x_maxs=[5, 10], y_maxs=[5, 10], metadata=meta)
        expected = DetectionAnnotation(
            x_mins=[pytest.approx(0), pytest.approx(1)],
            y_mins=[pytest.approx(0), pytest.approx(1)],
            x_maxs=[pytest.approx(1), pytest.approx(1)],
            y_maxs=[pytest.approx(1), pytest.approx(1)],
            metadata=meta
        )

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_clip_annotation_denormalized_boxes_with_size(self):
        config = [{'type': 'clip_boxes', 'apply_to': 'annotation', 'boxes_normalized': False, 'size': 10}]
        meta = {'image_size': (10, 10, 3)}
        annotation = DetectionAnnotation(x_mins=[-1, 9], y_mins=[0, 11], x_maxs=[5, 10], y_maxs=[5, 10], metadata=meta)
        expected = DetectionAnnotation(
            x_mins=[pytest.approx(0), pytest.approx(9)],
            y_mins=[pytest.approx(0), pytest.approx(10)],
            x_maxs=[pytest.approx(5), pytest.approx(10)],
            y_maxs=[pytest.approx(5), pytest.approx(10)],
            metadata=meta
        )

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_clip_annotation_normalized_boxes_with_size_as_normalized(self):
        config = [{'type': 'clip_boxes', 'apply_to': 'annotation', 'boxes_normalized': True, 'size': 10}]
        meta = {'image_size': (10, 10, 3)}
        annotation = DetectionAnnotation(x_mins=[-1, 9], y_mins=[0, 11], x_maxs=[5, 10], y_maxs=[5, 10], metadata=meta)
        expected = DetectionAnnotation(
            x_mins=[pytest.approx(0), pytest.approx(1)],
            y_mins=[pytest.approx(0), pytest.approx(1)],
            x_maxs=[pytest.approx(1), pytest.approx(1)],
            y_maxs=[pytest.approx(1), pytest.approx(1)],
            metadata=meta
        )

        postprocess_data(PostprocessingExecutor(config), [annotation], [None])

        assert annotation == expected

    def test_clip_prediction_denormalized_boxes(self):
        config = [{'type': 'clip_boxes', 'apply_to': 'prediction', 'boxes_normalized': False}]
        annotation = DetectionAnnotation(metadata={'image_size': (10, 10, 3)})
        prediction = DetectionPrediction(x_mins=[-1, 9], y_mins=[0, 11], x_maxs=[5, 10], y_maxs=[5, 10])
        expected_prediction = DetectionPrediction(
            x_mins=[pytest.approx(0), pytest.approx(9)],
            y_mins=[pytest.approx(0), pytest.approx(10)],
            x_maxs=[pytest.approx(5), pytest.approx(10)],
            y_maxs=[pytest.approx(5), pytest.approx(10)]
        )

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction

    def test_clip_prediction_normalized_boxes(self):
        config = [{'type': 'clip_boxes', 'apply_to': 'prediction', 'boxes_normalized': True}]
        annotation = DetectionAnnotation(metadata={'image_size': (10, 10, 3)})
        prediction = DetectionPrediction(x_mins=[-1, 9], y_mins=[0, 11], x_maxs=[5, 10], y_maxs=[5, 10])
        expected_prediction = DetectionPrediction(
            x_mins=[pytest.approx(0), pytest.approx(1)],
            y_mins=[pytest.approx(0), pytest.approx(1)],
            x_maxs=[pytest.approx(1), pytest.approx(1)],
            y_maxs=[pytest.approx(1), pytest.approx(1)]
        )

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction

    def test_clip_predictions_denormalized_boxes_with_size(self):
        config = [{'type': 'clip_boxes', 'apply_to': 'prediction', 'boxes_normalized': False, 'size': 10}]
        annotation = DetectionAnnotation(metadata={'image_size': (10, 10, 3)})
        prediction = DetectionPrediction(x_mins=[-1, 9], y_mins=[0, 11], x_maxs=[5, 10], y_maxs=[5, 10])
        expected_prediction = DetectionPrediction(
            x_mins=[pytest.approx(0), pytest.approx(9)],
            y_mins=[pytest.approx(0), pytest.approx(10)],
            x_maxs=[pytest.approx(5), pytest.approx(10)],
            y_maxs=[pytest.approx(5), pytest.approx(10)]
        )

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction

    def test_clip_predictions_normalized_boxes_with_size_as_normalized(self):
        config = [{'type': 'clip_boxes', 'apply_to': 'prediction', 'boxes_normalized': True, 'size': 10}]
        annotation = DetectionAnnotation(metadata={'image_size': (10, 10, 3)})
        prediction = DetectionPrediction(x_mins=[-1, 9], y_mins=[0, 11], x_maxs=[5, 10], y_maxs=[5, 10])
        expected_prediction = DetectionPrediction(
            x_mins=[pytest.approx(0), pytest.approx(1)],
            y_mins=[pytest.approx(0), pytest.approx(1)],
            x_maxs=[pytest.approx(1), pytest.approx(1)],
            y_maxs=[pytest.approx(1), pytest.approx(1)]
        )

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction

    def test_cast_to_int_default(self):
        config = [{'type': 'cast_to_int'}]
        annotation = DetectionAnnotation(x_mins=[-1, 9], y_mins=[0, 11], x_maxs=[5, 10], y_maxs=[5, 10])
        prediction = DetectionPrediction(
            x_mins=[-1.1, -9.9],
            y_mins=[0.5, 11.5],
            x_maxs=[5.9, 10.9],
            y_maxs=[5.1, 10.1]
        )
        expected_annotation = DetectionAnnotation(x_mins=[-1, 9], y_mins=[0, 11], x_maxs=[5, 10], y_maxs=[5, 10])
        expected_prediction = DetectionPrediction(x_mins=[-1, -10], y_mins=[0, 12], x_maxs=[6, 11], y_maxs=[5, 10])

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction and annotation == expected_annotation

    def test_cast_to_int_to_nearest(self):
        config = [{'type': 'cast_to_int', 'round_policy': 'nearest'}]
        annotation = DetectionAnnotation(x_mins=[-1, 9], y_mins=[0, 11], x_maxs=[5, 10], y_maxs=[5, 10])
        prediction = DetectionPrediction(
            x_mins=[-1.1, -9.9],
            y_mins=[0.5, 11.5],
            x_maxs=[5.9, 10.9],
            y_maxs=[5.1, 10.1]
        )
        expected_annotation = DetectionAnnotation(x_mins=[-1, 9], y_mins=[0, 11], x_maxs=[5, 10], y_maxs=[5, 10])
        expected_prediction = DetectionPrediction(x_mins=[-1, -10], y_mins=[0, 12], x_maxs=[6, 11], y_maxs=[5, 10])

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction and annotation == expected_annotation

    def test_cast_to_int_to_nearest_to_zero(self):
        config = [{'type': 'cast_to_int', 'round_policy': 'nearest_to_zero'}]
        annotation = DetectionAnnotation(x_mins=[-1, 9], y_mins=[0, 11], x_maxs=[5, 10], y_maxs=[5, 10])
        prediction = DetectionPrediction(
            x_mins=[-1.1, -9.9],
            y_mins=[0.5, 11.5],
            x_maxs=[5.9, 10.9],
            y_maxs=[5.1, 10.1]
        )
        expected_annotation = DetectionAnnotation(x_mins=[-1, 9], y_mins=[0, 11], x_maxs=[5, 10], y_maxs=[5, 10])
        expected_prediction = DetectionPrediction(x_mins=[-1, -9], y_mins=[0, 11], x_maxs=[5, 10], y_maxs=[5, 10])

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction and annotation == expected_annotation

    def test_cast_to_int_to_lower(self):
        config = [{'type': 'cast_to_int', 'round_policy': 'lower'}]
        annotation = DetectionAnnotation(x_mins=[-1, 9], y_mins=[0, 11], x_maxs=[5, 10], y_maxs=[5, 10])
        prediction = DetectionPrediction(
            x_mins=[-1.1, -9.9],
            y_mins=[0.5, 11.5],
            x_maxs=[5.9, 10.9],
            y_maxs=[5.1, 10.1]
        )
        expected_annotation = DetectionAnnotation(x_mins=[-1, 9], y_mins=[0, 11], x_maxs=[5, 10], y_maxs=[5, 10])
        expected_prediction = DetectionPrediction(x_mins=[-2, -10], y_mins=[0, 11], x_maxs=[5, 10], y_maxs=[5, 10])

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction and annotation == expected_annotation

    def test_cast_to_int_to_greater(self):
        config = [{'type': 'cast_to_int', 'round_policy': 'greater'}]
        annotation = DetectionAnnotation(x_mins=[-1, 9], y_mins=[0, 11], x_maxs=[5, 10], y_maxs=[5, 10])
        prediction = DetectionPrediction(
            x_mins=[-1.1, -9.9],
            y_mins=[0.5, 11.5],
            x_maxs=[5.9, 10.9],
            y_maxs=[5.1, 10.1]
        )
        expected_annotation = DetectionAnnotation(x_mins=[-1, 9], y_mins=[0, 11], x_maxs=[5, 10], y_maxs=[5, 10])
        expected_prediction = DetectionPrediction(x_mins=[-1, -9], y_mins=[1, 12], x_maxs=[6, 11], y_maxs=[6, 11])

        postprocess_data(PostprocessingExecutor(config), [annotation], [prediction])

        assert prediction == expected_prediction and annotation == expected_annotation

    def test_cast_to_int_to_unknown_raise_config_error(self):
        config = [{'type': 'cast_to_int', 'round_policy': 'unknown'}]

        with pytest.raises(ConfigError):
            postprocess_data(PostprocessingExecutor(config), [None], [None])
