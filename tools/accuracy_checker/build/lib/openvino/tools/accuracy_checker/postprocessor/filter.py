"""
Copyright (c) 2018-2022 Intel Corporation

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

from functools import singledispatch
import numpy as np

from ..config import BaseField, BoolField
from ..dependency import ClassProvider
from ..postprocessor.postprocessor import PostprocessorWithSpecificTargets
from ..representation import (DetectionAnnotation, DetectionPrediction, TextDetectionAnnotation,
                              TextDetectionPrediction, PoseEstimationPrediction, PoseEstimationAnnotation)
from ..utils import in_interval, polygon_from_points, convert_to_range


class FilterPostprocessor(PostprocessorWithSpecificTargets):
    __provider__ = 'filter'

    annotation_types = (DetectionAnnotation, TextDetectionAnnotation)
    prediction_types = (DetectionPrediction, TextDetectionPrediction)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'remove_filtered': BoolField(
                optional=True, default=False,
                description="Removing filtered data. Annotations support ignoring filtered data without removing"
                            " as default,in other cases filtered data will be removed automatically."
            )
        })

        for functor in BaseFilter.providers:
            parameters[functor] = BaseField(optional=True, description=functor)

        return parameters

    def __init__(self, *args, **kwargs):
        self._filters = []
        self.remove_filtered = False
        super().__init__(*args, **kwargs)

    def configure(self):
        config = self.config.copy()
        config.pop('type')
        self.remove_filtered = self.get_value_from_config('remove_filtered')
        config.pop('remove_filtered', False)
        config.pop('annotation_source', None)
        config.pop('prediction_source', None)
        config.pop('apply_to', None)

        for key, value in config.items():
            self._filters.append(BaseFilter.provide(key, value))

    def process_image(self, annotation, prediction):
        for functor in self._filters:
            for target in annotation:
                self._filter_entry_by(target, functor)

            for target in prediction:
                self._filter_entry_by(target, functor)

        return annotation, prediction

    def _filter_entry_by(self, entry, functor):
        ignored_key = 'difficult_boxes'

        if not self.remove_filtered and isinstance(entry, (DetectionAnnotation, DetectionPrediction,
                                                           TextDetectionAnnotation, TextDetectionPrediction,
                                                           PoseEstimationAnnotation, PoseEstimationPrediction)):
            ignored = entry.metadata.setdefault(ignored_key, [])
            ignored.extend(functor(entry))
        else:
            entry.remove(functor(entry))

        return entry


class BaseFilter(ClassProvider):
    __provider_type__ = 'filter'

    def __init__(self, filter_arg):
        self.filter_arg = filter_arg

    def __call__(self, entry):
        return self.apply_filter(entry, self.filter_arg)

    def apply_filter(self, entry, filter_arg):
        raise NotImplementedError


class FilterByLabels(BaseFilter):
    __provider__ = 'labels'

    def apply_filter(self, entry, filter_arg):
        filtered = []
        for index, label in enumerate(entry.labels):
            if label in filter_arg:
                filtered.append(index)

        return filtered


class FilterByMinConfidence(BaseFilter):
    __provider__ = 'min_confidence'

    def apply_filter(self, entry, filter_arg):
        filtered = []

        if isinstance(entry, DetectionAnnotation):
            return filtered

        for index, score in enumerate(entry.scores):
            if score < filter_arg:
                filtered.append(index)

        return filtered


class FilterTopK(BaseFilter):
    __provider__ = 'top_k'

    def apply_filter(self, entry, filter_arg):
        filtered = []

        if isinstance(entry, DetectionAnnotation):
            return filtered

        if len(entry.scores) <= filter_arg:
            return filtered
        scores_inds = np.argsort(entry.scores)[::-1]
        non_filtered = scores_inds[:int(filter_arg)]

        return [ind for ind in range(len(entry.scores)) if ind not in non_filtered]

class FilterByHeightRange(BaseFilter):
    __provider__ = 'height_range'

    annotation_types = (DetectionAnnotation, TextDetectionAnnotation)
    prediction_types = (DetectionPrediction, TextDetectionPrediction)

    def apply_filter(self, entry, filter_arg):
        @singledispatch
        def filter_func(entry_value, height_range_):
            return []

        @filter_func.register(DetectionAnnotation)
        @filter_func.register(DetectionPrediction)
        def _(entry_value, height_range_):
            filtered = []
            for index, (y_min, y_max) in enumerate(zip(entry_value.y_mins, entry_value.y_maxs)):
                height = y_max - y_min
                if not in_interval(height, height_range_):
                    filtered.append(index)

            return filtered

        @filter_func.register(TextDetectionAnnotation)
        @filter_func.register(TextDetectionPrediction)
        def _(entry_values, height_range_):
            filtered = []
            for index, polygon_points in enumerate(entry_values.points):
                left_bottom_point, left_top_point, right_top_point, right_bottom_point = polygon_points
                left_side_height = np.linalg.norm(left_bottom_point - left_top_point)
                right_side_height = np.linalg.norm(right_bottom_point - right_top_point)
                if not in_interval(np.mean([left_side_height, right_side_height]), height_range_):
                    filtered.append(index)

            return filtered

        return filter_func(entry, convert_to_range(filter_arg))


class FilterByWidthRange(BaseFilter):
    __provider__ = 'width_range'

    annotation_types = (DetectionAnnotation, TextDetectionAnnotation)
    prediction_types = (DetectionPrediction, TextDetectionPrediction)

    def apply_filter(self, entry, filter_arg):
        @singledispatch
        def filter_func(entry_value, width_range_):
            return []

        @filter_func.register(DetectionAnnotation)
        @filter_func.register(DetectionPrediction)
        def _(entry_value, width_range_):
            filtered = []
            for index, (x_min, x_max) in enumerate(zip(entry_value.x_mins, entry_value.x_maxs)):
                width = x_max - x_min
                if not in_interval(width, width_range_):
                    filtered.append(index)

            return filtered

        @filter_func.register(TextDetectionAnnotation)
        @filter_func.register(TextDetectionPrediction)
        def _(entry_values, width_range_):
            filtered = []
            for index, polygon_points in enumerate(entry_values.points):
                left_bottom_point, left_top_point, right_top_point, right_bottom_point = polygon_points
                top_width = np.linalg.norm(right_top_point - left_top_point)
                bottom_width = np.linalg.norm(right_bottom_point - left_bottom_point)
                if not in_interval(top_width, width_range_) or not in_interval(bottom_width, width_range_):
                    filtered.append(index)

            return filtered

        return filter_func(entry, convert_to_range(filter_arg))


class FilterByAreaRange(BaseFilter):
    __provider__ = 'area_range'

    annotation_types = (TextDetectionAnnotation, PoseEstimationAnnotation)
    prediction_types = (TextDetectionPrediction, )

    def apply_filter(self, entry, filter_arg):
        area_range = convert_to_range(filter_arg)

        @singledispatch
        def filter_func(entry, area_range):
            return []

        @filter_func.register(PoseEstimationAnnotation)
        @filter_func.register(PoseEstimationPrediction)
        def _(entry, area_range):
            filtered = []
            areas = entry.areas
            for area_id, area in enumerate(areas):
                if not in_interval(area, area_range):
                    filtered.append(area_id)
            return filtered

        @filter_func.register(TextDetectionAnnotation)
        @filter_func.register(TextDetectionPrediction)
        def _(entry, area_range):
            filtered = []
            for index, polygon_points in enumerate(entry.points):
                if not in_interval(polygon_from_points(polygon_points).area, area_range):
                    filtered.append(index)
            return filtered

        return filter_func(entry, area_range)


class FilterEmpty(BaseFilter):
    __provider__ = 'is_empty'

    def apply_filter(self, entry: DetectionAnnotation, filter_arg):
        return np.where(np.bitwise_or(entry.x_maxs - entry.x_mins <= 0, entry.y_maxs - entry.y_mins <= 0))[0]


class FilterByVisibility(BaseFilter):
    __provider__ = 'min_visibility'

    _VISIBILITY_LEVELS = {
        'heavy occluded': 0,
        'partially occluded': 1,
        'visible': 2
    }

    def apply_filter(self, entry, filter_arg):
        filtered = []
        min_visibility_level = self.visibility_level(filter_arg)
        for index, visibility in enumerate(entry.metadata.get('visibilities', [])):
            if self.visibility_level(visibility) < min_visibility_level:
                filtered.append(index)

        return filtered

    def visibility_level(self, visibility):
        level = self._VISIBILITY_LEVELS.get(visibility)
        if level is None:
            message = 'Unknown visibility level "{}". Supported only "{}"'
            raise ValueError(message.format(visibility, ','.join(self._VISIBILITY_LEVELS.keys())))

        return level


class FilterByAspectRatio(BaseFilter):
    __provider__ = 'aspect_ratio'

    def apply_filter(self, entry, filter_arg):
        aspect_ratio = convert_to_range(filter_arg)

        filtered = []
        coordinates = zip(entry.x_mins, entry.y_mins, entry.x_maxs, entry.y_maxs)
        for index, (x_min, y_min, x_max, y_max) in enumerate(coordinates):
            ratio = (y_max - y_min) / np.maximum(x_max - x_min, np.finfo(np.float64).eps)
            if not in_interval(ratio, aspect_ratio):
                filtered.append(index)

        return filtered


class FilterByAreaRatio(BaseFilter):
    __provider__ = 'area_ratio'

    def apply_filter(self, entry, filter_arg):
        area_ratio = convert_to_range(filter_arg)

        filtered = []
        if not isinstance(entry, DetectionAnnotation):
            return filtered

        image_size = entry.metadata.get('image_size')
        if not image_size:
            return filtered
        image_size = image_size[0]

        image_area = image_size[0] * image_size[1]

        occluded_indices = entry.metadata.get('is_occluded', [])
        coordinates = zip(entry.x_mins, entry.y_mins, entry.x_maxs, entry.y_maxs)
        for index, (x_min, y_min, x_max, y_max) in enumerate(coordinates):
            width, height = x_max - x_min, y_max - y_min
            area = np.sqrt(float(width * height) / np.maximum(image_area, np.finfo(np.float64).eps))
            if not in_interval(area, area_ratio) or index in occluded_indices:
                filtered.append(index)

        return filtered


class FilterInvalidBoxes(BaseFilter):
    __provider__ = 'invalid_boxes'

    def apply_filter(self, entry, filter_arg):
        infinite_mask_x = np.logical_or(~np.isfinite(entry.x_mins), ~np.isfinite(entry.x_maxs)) # pylint: disable=E1130
        infinite_mask_y = np.logical_or(~np.isfinite(entry.y_mins), ~np.isfinite(entry.y_maxs)) # pylint: disable=E1130
        infinite_mask = np.logical_or(infinite_mask_x, infinite_mask_y)

        return np.argwhere(infinite_mask).reshape(-1).tolist()
