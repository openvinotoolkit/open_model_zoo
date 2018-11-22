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

from ..config import BaseField, BoolField
from ..dependency import ClassProvider
from ..postprocessor.postprocessor import PostprocessorWithSpecificTargets, PostprocessorWithTargetsConfigValidator
from ..representation import DetectionAnnotation, DetectionPrediction
from ..utils import (string_to_tuple, in_interval)


class FilterConfig(PostprocessorWithTargetsConfigValidator):
    remove_filtered = BoolField(optional=True)

    def __init__(self, config_uri, **kwargs):
        super().__init__(config_uri, **kwargs)
        for filter_ in BaseFilter.providers:
            self._fields[filter_] = BaseField(optional=True)


class FilterPostprocessor(PostprocessorWithSpecificTargets):
    __provider__ = 'filter'
    annotation_types = (DetectionAnnotation, )
    prediction_types = (DetectionPrediction, )

    def __init__(self, *args, **kwargs):
        self._filters = []
        self.remove_filtered = False
        super().__init__(*args, **kwargs)

    def validate_config(self):
        filter_config = FilterConfig(self.__provider__, on_extra_argument=FilterConfig.ERROR_ON_EXTRA_ARGUMENT)
        filter_config.validate(self.config)

    def configure(self):
        config = self.config.copy()
        config.pop('type')
        self.remove_filtered = config.pop('remove_filtered', False)
        config.pop('annotation_source', None)
        config.pop('prediction_source', None)
        config.pop('apply_to', None)

        for k, v in config.items():
            self._filters.append(BaseFilter.provide(k, v))

    def process_image(self, annotation, prediction):
        for functor in self._filters:
            for target in annotation:
                self._filter_entry_by(target, functor)

            for target in prediction:
                self._filter_entry_by(target, functor)

        return annotation, prediction

    def _filter_entry_by(self, entry, functor):
        ignored_key = 'difficult_boxes'

        if not self.remove_filtered and isinstance(entry, (DetectionAnnotation, DetectionPrediction)):
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

    def apply_filter(self, entry, labels):
        filtered = []
        for index, label in enumerate(entry.labels):
            if label in labels:
                filtered.append(index)

        return filtered


class FilterByMinConfidence(BaseFilter):
    __provider__ = 'min_confidence'

    def apply_filter(self, entry, min_confidence):
        if isinstance(entry, DetectionAnnotation):
            return []

        filtered = []
        for index, score in enumerate(entry.scores):
            if score < min_confidence:
                filtered.append(index)

        return filtered


class FilterByHeightRange(BaseFilter):
    __provider__ = 'height_range'

    def apply_filter(self, entry, height_range):
        if isinstance(height_range, str):
            height_range = string_to_tuple(height_range)
        elif not isinstance(height_range, tuple) and not isinstance(height_range, list):
            height_range = [height_range]

        filtered = []
        for index, (y_min, y_max) in enumerate(zip(entry.y_mins, entry.y_maxs)):
            height = y_max - y_min
            if not in_interval(height, height_range):
                filtered.append(index)
        return filtered


class FilterByWidthRange(BaseFilter):
    __provider__ = 'width_range'

    def apply_filter(self, entry, width_range):
        if isinstance(width_range, str):
            width_range = string_to_tuple(width_range)
        elif not isinstance(width_range, tuple) and not isinstance(width_range, list):
            width_range = [width_range]

        filtered = []
        for index, (x_min, x_max) in enumerate(zip(entry.x_mins, entry.x_maxs)):
            width = x_max - x_min
            if not in_interval(width, width_range):
                filtered.append(index)

        return filtered


class FilterEmpty(BaseFilter):
    __provider__ = 'is_empty'

    def apply_filter(self, entry: DetectionAnnotation, is_empty):
        idx = np.bitwise_or(entry.x_maxs - entry.x_mins <= 0,
                            entry.y_maxs - entry.y_mins <= 0)

        filtered = np.where(idx)[0]

        return filtered


class FilterByVisibility(BaseFilter):
    __provider__ = 'min_visibility'

    _VISIBILITY_LEVELS = {
        'heavy occluded': 0,
        'partially occluded': 1,
    }

    def apply_filter(self, entry, min_visibility):
        filtered = []
        if not isinstance(entry, DetectionAnnotation):
            return filtered

        min_visibility_level = self.visibility_level(min_visibility)
        for index, visibility in enumerate(entry.metadata.get('visibilities', [])):
            if self.visibility_level(visibility) < min_visibility_level:
                filtered.append(index)

        return filtered

    def visibility_level(self, visibility=None):
        max_level = max(self._VISIBILITY_LEVELS.values()) + 1
        if visibility is None:
            return max_level

        value = ' '.join(visibility.lower().split())
        level = self._VISIBILITY_LEVELS.get(value)
        return level if level is not None else max_level


class FilterByAspectRatio(BaseFilter):
    __provider__ = 'aspect_ratio'

    def apply_filter(self, entry, aspect_ratio):
        if isinstance(aspect_ratio, str):
            aspect_ratio = string_to_tuple(aspect_ratio)
        elif not isinstance(aspect_ratio, tuple) and not isinstance(aspect_ratio, list):
            aspect_ratio = [aspect_ratio]

        filtered = []
        for index, (x_min, y_min, x_max, y_max) in enumerate(zip(entry.x_mins, entry.y_mins,
                                                                 entry.x_maxs, entry.y_maxs)):
            ratio = (y_max - y_min) / (x_max - x_min)
            if not in_interval(ratio, aspect_ratio):
                filtered.append(index)

        return filtered


class FilterByAreaRatio(BaseFilter):
    __provider__ = 'area_ratio'

    def apply_filter(self, entry, area_ratio):
        if isinstance(area_ratio, str):
            area_ratio = string_to_tuple(area_ratio)
        elif not isinstance(area_ratio, tuple) and not isinstance(area_ratio, list):
            area_ratio = [area_ratio]

        filtered = []
        if not isinstance(entry, DetectionAnnotation):
            return filtered

        image_size = entry.metadata.get('image_size')
        if not image_size:
            return filtered
        image_area = image_size[0] * image_size[1]

        occluded_indices = entry.metadata.get('is_occluded', [])
        for index, (x_min, y_min, x_max, y_max) in enumerate(zip(entry.x_mins, entry.y_mins,
                                                                 entry.x_maxs, entry.y_maxs)):
            width = x_max - x_min
            height = y_max - y_min
            area = np.sqrt(float(width * height) / image_area)

            is_occluded = index in occluded_indices
            if not in_interval(area, area_ratio) or is_occluded:
                filtered.append(index)

        return filtered
