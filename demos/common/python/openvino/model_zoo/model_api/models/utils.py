"""
 Copyright (C) 2020-2022 Intel Corporation

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

import cv2
import math
import inspect
import typing
import itertools
from functools import wraps
from collections.abc import Sequence
import numpy as np
from numpy import floating

class Detection:
    def __init__(self, xmin, ymin, xmax, ymax, score, id):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.score = score
        self.id = id

    def bottom_left_point(self):
        return self.xmin, self.ymin

    def top_right_point(self):
        return self.xmax, self.ymax

    def get_coords(self):
        return self.xmin, self.ymin, self.xmax, self.ymax


def clip_detections(detections, size):
    for detection in detections:
        detection.xmin = max(int(detection.xmin), 0)
        detection.ymin = max(int(detection.ymin), 0)
        detection.xmax = min(int(detection.xmax), size[1])
        detection.ymax = min(int(detection.ymax), size[0])
    return detections


class DetectionWithLandmarks(Detection):
    def __init__(self, xmin, ymin, xmax, ymax, score, id, landmarks_x, landmarks_y):
        super().__init__(xmin, ymin, xmax, ymax, score, id)
        self.landmarks = []
        for x, y in zip(landmarks_x, landmarks_y):
            self.landmarks.append((x, y))


class OutputTransform:
    def __init__(self, input_size, output_resolution):
        self.output_resolution = output_resolution
        if self.output_resolution:
            self.new_resolution = self.compute_resolution(input_size)

    def compute_resolution(self, input_size):
        self.input_size = input_size
        size = self.input_size[::-1]
        self.scale_factor = min(self.output_resolution[0] / size[0],
                                self.output_resolution[1] / size[1])
        return self.scale(size)

    def resize(self, image):
        if not self.output_resolution:
            return image
        curr_size = image.shape[:2]
        if curr_size != self.input_size:
            self.new_resolution = self.compute_resolution(curr_size)
        if self.scale_factor == 1:
            return image
        return cv2.resize(image, self.new_resolution)

    def scale(self, inputs):
        if not self.output_resolution or self.scale_factor == 1:
            return inputs
        return (np.array(inputs) * self.scale_factor).astype(np.int32)


class InputTransform:
    def __init__(self, reverse_input_channels=False, mean_values=None, scale_values=None):
        self.reverse_input_channels = reverse_input_channels
        self.is_trivial = not (reverse_input_channels or mean_values or scale_values)
        self.means = np.array(mean_values, dtype=np.float32) if mean_values else np.array([0., 0., 0.])
        self.std_scales = np.array(scale_values, dtype=np.float32) if scale_values else np.array([1., 1., 1.])

    def __call__(self, inputs):
        if self.is_trivial:
            return inputs
        if self.reverse_input_channels:
            inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB)
        return (inputs - self.means) / self.std_scales


def load_labels(label_file):
    with open(label_file, 'r') as f:
        labels_map = [x.strip() for x in f]
    return labels_map

def create_hard_prediction_from_soft_prediction(
    soft_prediction: np.ndarray, soft_threshold: float, blur_strength: int = 5
) -> np.ndarray:
    """
    Creates a hard prediction containing the final label index per pixel
    :param soft_prediction: Output from segmentation network. Assumes floating point
                            values, between 0.0 and 1.0. Can be a 2d-array of shape
                            (height, width) or per-class segmentation logits of shape
                            (height, width, num_classes)
    :param soft_threshold: minimum class confidence for each pixel.
                            The higher the value, the more strict the segmentation is
                            (usually set to 0.5)
    :param blur_strength: The higher the value, the smoother the segmentation output
                            will be, but less accurate
    :return: Numpy array of the hard prediction
    """
    soft_prediction_blurred = cv2.blur(soft_prediction, (blur_strength, blur_strength))
    if len(soft_prediction.shape) == 3:
        # Apply threshold to filter out `unconfident` predictions, then get max along
        # class dimension
        soft_prediction_blurred[soft_prediction_blurred < soft_threshold] = 0
        hard_prediction = np.argmax(soft_prediction_blurred, axis=2)
    elif len(soft_prediction.shape) == 2:
        # In the binary case, simply apply threshold
        hard_prediction = soft_prediction_blurred > soft_threshold
    else:
        raise ValueError(
            f"Invalid prediction input of shape {soft_prediction.shape}. "
            f"Expected either a 2D or 3D array."
        )
    return hard_prediction

def get_bases(parameter) -> set:
    """Function to get set of all base classes of parameter"""

    def __get_bases(parameter_type):
        return [parameter_type.__name__] + list(
            itertools.chain.from_iterable(
                __get_bases(t1) for t1 in parameter_type.__bases__
            )
        )

    return set(__get_bases(type(parameter)))


def get_parameter_repr(parameter) -> str:
    """Function to get parameter representation"""
    try:
        parameter_str = repr(parameter)
    # pylint: disable=broad-except
    except Exception:
        parameter_str = "<unable to get parameter repr>"
    return parameter_str


def raise_value_error_if_parameter_has_unexpected_type(
    parameter, parameter_name, expected_type
):
    """Function raises ValueError exception if parameter has unexpected type"""
    if isinstance(expected_type, typing.ForwardRef):
        expected_type = expected_type.__forward_arg__
    if isinstance(expected_type, str):
        parameter_types = get_bases(parameter)
        if not any(t == expected_type for t in parameter_types):
            parameter_str = get_parameter_repr(parameter)
            raise ValueError(
                f"Unexpected type of '{parameter_name}' parameter, expected: {expected_type}, "
                f"actual value: {parameter_str}"
            )
        return
    if expected_type == float:
        expected_type = (int, float, floating)
    if not isinstance(parameter, expected_type):
        parameter_type = type(parameter)
        parameter_str = get_parameter_repr(parameter)
        raise ValueError(
            f"Unexpected type of '{parameter_name}' parameter, expected: {expected_type}, actual: {parameter_type}, "
            f"actual value: {parameter_str}"
        )


def check_nested_elements_type(iterable, parameter_name, expected_type):
    """Function raises ValueError exception if one of elements in collection has unexpected type"""
    for element in iterable:
        check_parameter_type(
            parameter=element,
            parameter_name=f"nested {parameter_name}",
            expected_type=expected_type,
        )


def check_dictionary_keys_values_type(
    parameter, parameter_name, expected_key_class, expected_value_class
):
    """Function raises ValueError exception if dictionary key or value has unexpected type"""
    for key, value in parameter.items():
        check_parameter_type(
            parameter=key,
            parameter_name=f"key in {parameter_name}",
            expected_type=expected_key_class,
        )
        check_parameter_type(
            parameter=value,
            parameter_name=f"value in {parameter_name}",
            expected_type=expected_value_class,
        )


def check_nested_classes_parameters(
    parameter, parameter_name, origin_class, nested_elements_class
):
    """Function to check type of parameters with nested elements"""
    # Checking origin class
    raise_value_error_if_parameter_has_unexpected_type(
        parameter=parameter, parameter_name=parameter_name, expected_type=origin_class
    )
    # Checking nested elements
    if origin_class == dict:
        if len(nested_elements_class) != 2:
            raise TypeError(
                "length of nested expected types for dictionary should be equal to 2"
            )
        key, value = nested_elements_class
        check_dictionary_keys_values_type(
            parameter=parameter,
            parameter_name=parameter_name,
            expected_key_class=key,
            expected_value_class=value,
        )
    if origin_class in [list, set, tuple, Sequence]:
        if origin_class == tuple:
            tuple_length = len(nested_elements_class)
            if tuple_length > 2:
                raise NotImplementedError(
                    "length of nested expected types for Tuple should not exceed 2"
                )
            if tuple_length == 2:
                if nested_elements_class[1] != Ellipsis:
                    raise NotImplementedError("expected homogeneous tuple annotation")
                nested_elements_class = nested_elements_class[0]
        else:
            if len(nested_elements_class) != 1:
                raise TypeError(
                    "length of nested expected types for Sequence should be equal to 1"
                )
        check_nested_elements_type(
            iterable=parameter,
            parameter_name=parameter_name,
            expected_type=nested_elements_class,
        )


def check_parameter_type(parameter, parameter_name, expected_type):
    """Function extracts nested expected types and raises ValueError exception if parameter has unexpected type"""
    # pylint: disable=W0212
    if expected_type in [typing.Any, inspect._empty]:  # type: ignore
        return
    if not isinstance(expected_type, typing._GenericAlias):  # type: ignore
        raise_value_error_if_parameter_has_unexpected_type(
            parameter=parameter,
            parameter_name=parameter_name,
            expected_type=expected_type,
        )
        return
    expected_type_dict = expected_type.__dict__
    origin_class = expected_type_dict.get("__origin__")
    nested_elements_class = expected_type_dict.get("__args__")
    # Union type with nested elements check
    if origin_class == typing.Union:
        expected_args = expected_type_dict.get("__args__")
        checks_counter = 0
        errors_counter = 0
        for expected_arg in expected_args:
            try:
                checks_counter += 1
                check_parameter_type(parameter, parameter_name, expected_arg)
            except ValueError:
                errors_counter += 1
        if errors_counter == checks_counter:
            actual_type = type(parameter)
            raise ValueError(
                f"Unexpected type of '{parameter_name}' parameter, expected: {expected_args}, "
                f"actual type: {actual_type}, actual value: {parameter}"
            )
    # Checking parameters with nested elements
    elif issubclass(origin_class, typing.Iterable):
        check_nested_classes_parameters(
            parameter=parameter,
            parameter_name=parameter_name,
            origin_class=origin_class,
            nested_elements_class=nested_elements_class,
        )


def check_input_parameters_type(custom_checks: typing.Optional[dict] = None):
    """
    Decorator to check input parameters type
    :param custom_checks: dictionary where key - name of parameter and value - custom check class
    """
    if custom_checks is None:
        custom_checks = {}

    def _check_input_parameters_type(function):
        @wraps(function)
        def validate(*args, **kwargs):
            # Forming expected types dictionary
            signature = inspect.signature(function)
            expected_types_map = signature.parameters
            if len(expected_types_map) < len(args):
                raise TypeError("Too many positional arguments")
            # Forming input parameters dictionary
            input_parameters_values_map = dict(zip(signature.parameters.keys(), args))
            for key, value in kwargs.items():
                if key in input_parameters_values_map:
                    raise TypeError(
                        f"Duplication of the parameter {key} -- both in args and kwargs"
                    )
                input_parameters_values_map[key] = value
            # Checking input parameters type
            for parameter_name in expected_types_map:
                parameter = input_parameters_values_map.get(parameter_name)
                if parameter_name not in input_parameters_values_map:
                    default_value = expected_types_map.get(parameter_name).default
                    # pylint: disable=protected-access
                    if default_value != inspect._empty:  # type: ignore
                        parameter = default_value
                if parameter_name in custom_checks:
                    custom_check = custom_checks[parameter_name]
                    if custom_check is None:
                        continue
                    custom_check(parameter, parameter_name).check()
                else:
                    check_parameter_type(
                        parameter=parameter,
                        parameter_name=parameter_name,
                        expected_type=expected_types_map.get(parameter_name).annotation,
                    )
            return function(**input_parameters_values_map)

        return validate

    return _check_input_parameters_type

def resize_image(image, size, keep_aspect_ratio=False, interpolation=cv2.INTER_LINEAR):
    if not keep_aspect_ratio:
        resized_frame = cv2.resize(image, size, interpolation=interpolation)
    else:
        h, w = image.shape[:2]
        scale = min(size[1] / h, size[0] / w)
        resized_frame = cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)
    return resized_frame


def resize_image_with_aspect(image, size, interpolation=cv2.INTER_LINEAR):
    return resize_image(image, size, keep_aspect_ratio=True, interpolation=interpolation)


def pad_image(image, size):
    h, w = image.shape[:2]
    if h != size[1] or w != size[0]:
        image = np.pad(image, ((0, size[1] - h), (0, size[0] - w), (0, 0)),
                               mode='constant', constant_values=0)
    return image


def resize_image_letterbox(image, size, interpolation=cv2.INTER_LINEAR):
    ih, iw = image.shape[0:2]
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = cv2.resize(image, (nw, nh), interpolation=interpolation)
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    resized_image = np.pad(image, ((dy, dy + (h - nh) % 2), (dx, dx + (w - nw) % 2), (0, 0)),
                           mode='constant', constant_values=128)
    return resized_image


def crop_resize(image, size):
    desired_aspect_ratio = size[1] / size[0] # width / height
    if desired_aspect_ratio == 1:
        if (image.shape[0] > image.shape[1]):
            offset = (image.shape[0] - image.shape[1]) // 2
            cropped_frame = image[offset:image.shape[1] + offset]
        else:
            offset = (image.shape[1] - image.shape[0]) // 2
            cropped_frame = image[:, offset:image.shape[0] + offset]
    elif desired_aspect_ratio < 1:
        new_width = math.floor(image.shape[0] * desired_aspect_ratio)
        offset = (image.shape[1] - new_width) // 2
        cropped_frame = image[:, offset:new_width + offset]
    elif desired_aspect_ratio > 1:
        new_height = math.floor(image.shape[1] / desired_aspect_ratio)
        offset = (image.shape[0] - new_height) // 2
        cropped_frame = image[offset:new_height + offset]

    return cv2.resize(cropped_frame, size)


RESIZE_TYPES = {
    'crop' : crop_resize,
    'standard': resize_image,
    'fit_to_window': resize_image_with_aspect,
    'fit_to_window_letterbox': resize_image_letterbox,
}


INTERPOLATION_TYPES = {
    'LINEAR': cv2.INTER_LINEAR,
    'CUBIC': cv2.INTER_CUBIC,
    'NEAREST': cv2.INTER_NEAREST,
    'AREA': cv2.INTER_AREA,
}


def nms(x1, y1, x2, y2, scores, thresh, include_boundaries=False, keep_top_k=None):
    b = 1 if include_boundaries else 0
    areas = (x2 - x1 + b) * (y2 - y1 + b)
    order = scores.argsort()[::-1]

    if keep_top_k:
        order = order[:keep_top_k]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + b)
        h = np.maximum(0.0, yy2 - yy1 + b)
        intersection = w * h

        union = (areas[i] + areas[order[1:]] - intersection)
        overlap = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)

        order = order[np.where(overlap <= thresh)[0] + 1]

    return keep


def softmax(logits, axis=None, keepdims=False):
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=axis, keepdims=keepdims)
