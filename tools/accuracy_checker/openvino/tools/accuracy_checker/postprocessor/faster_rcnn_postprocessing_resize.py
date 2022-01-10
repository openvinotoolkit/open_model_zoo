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

from ..representation import DetectionPrediction, DetectionAnnotation
from ..postprocessor.postprocessor import Postprocessor
from ..utils import get_size_from_config
from ..config import NumberField, BoolField


class FRCNNPostprocessingBboxResize(Postprocessor):
    """
    Resize normalized predicted bounding boxes coordinates (i.e. from [0, 1] range) to input image shape.
    """

    __provider__ = 'faster_rcnn_postprocessing_resize'

    prediction_types = (DetectionPrediction, )
    annotation_types = (DetectionAnnotation, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'dst_width': NumberField(value_type=int, optional=True, min_value=1, description="Destination width."),
            'dst_height': NumberField(value_type=int, optional=True, min_value=1, description="Destination height."),
            'size': NumberField(value_type=int, optional=True, min_value=1, description="Destination size."),
            'rescale': BoolField(optional=True, default=False)
        })
        return parameters

    def configure(self):
        self.dst_height, self.dst_width = get_size_from_config(self.config, allow_none=True)
        self.rescale = self.get_value_from_config('rescale')

    @staticmethod
    def get_coeff_x_y_from_metadata(image_metadata, rescale=False):
        assert image_metadata and 'geometric_operations' in image_metadata
        geometric_operations = image_metadata['geometric_operations'] # should be list of GeometricOperationMetadata

        if len(geometric_operations) == 1:
            op_name, op_props = geometric_operations[0]
            assert op_name == "resize", "Unknown case: one geometric operation with name '{}'".format(op_name)

            coeff_x = op_props['original_width']
            coeff_y = op_props['original_height']

        elif len(geometric_operations) == 2:
            op_name1, op_props1 = geometric_operations[0]
            op_name2, op_props2 = geometric_operations[1]
            assert op_name1 == "resize", (
                "Unknown case: two geometric operations, the first with name '{}'".format(op_name1))
            assert op_name2 == "padding", (
                "Unknown case: two geometric operations, the second with name '{}'".format(op_name2))
            pad = op_props2['pad']
            assert pad[0] == 0 and pad[1] == 0, "Unknown case: padding is not right_bottom"
            pad_pref_width = op_props2["pref_width"]
            pad_pref_height = op_props2["pref_height"]
            resize_scale_x = op_props1["scale_x"]
            resize_scale_y = op_props1["scale_y"]

            coeff_x = pad_pref_width / resize_scale_x
            coeff_y = pad_pref_height / resize_scale_y

        else:
            raise RuntimeError(
                "Unknown case: ""len(image_metadata['geometric_operations']) = {}".format(len(geometric_operations))
            )
        if rescale:
            input_shape = [value for value in image_metadata['input_shape'].values() if len(value) == 4]
            assert len(input_shape) == 1, 'suitable input shape not found or multi inputs detected'
            input_shape = input_shape[0]
            input_h, input_w = input_shape[2:] if input_shape[1] in [1, 3] else input_shape[1:3]
            coeff_x /= input_w
            coeff_y /= input_h
        return coeff_x, coeff_y

    def process_image(self, annotation, prediction):
        raise RuntimeError("Since `process_image_with_metadata` is overridden, this method MUST NOT be called")

    def process_image_with_metadata(self, annotation, prediction, image_metadata=None):
        assert image_metadata and 'geometric_operations' in image_metadata, (
            "Postprocessing step `faster_rcnn_postprocessing_resize` cannot work without "
            "metadata with `geometric_operations` field")
        coeff_x, coeff_y = self.get_coeff_x_y_from_metadata(image_metadata, self.rescale)

        for pred in prediction:
            pred.x_mins *= coeff_x
            pred.x_maxs *= coeff_x
            pred.y_mins *= coeff_y
            pred.y_maxs *= coeff_y

        return annotation, prediction
