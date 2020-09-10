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

from ..representation import DetectionPrediction, DetectionAnnotation
from ..postprocessor.postprocessor import Postprocessor
from ..utils import get_size_from_config
from ..config import NumberField


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
            'size': NumberField(value_type=int, optional=True, min_value=1, description="Destination size.")
        })
        return parameters

    def configure(self):
        self.dst_height, self.dst_width = get_size_from_config(self.config, allow_none=True)

    @staticmethod
    def get_coeff_x_y_from_metadata(image_metadata):
        assert image_metadata and 'geometric_operations' in image_metadata
        geometric_operations = image_metadata['geometric_operations'] # should be list of GeometricOperationMetadata

        if len(geometric_operations) == 1:
            op_name, op_props = geometric_operations[0]
            assert op_name == "resize", "Unknown case: one geometric operation with name '{}'".format(op_name)

            coeff_x = op_props['original_width']
            coeff_y = op_props['original_height']
            return coeff_x, coeff_y

        if len(geometric_operations) == 2:
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
            return coeff_x, coeff_y

        raise RuntimeError("Unknown case: "
                           "len(image_metadata['geometric_operations']) = {}".format(
                               len(geometric_operations)))

    def process_image(self, annotations, predictions):
        raise RuntimeError("Since `process_image_with_metadata` is overriden, this method MUST NOT be called")

    def process_image_with_metadata(self, annotations, predictions, image_metadata=None):
        assert image_metadata and 'geometric_operations' in image_metadata, (
            "Postprocessing step `faster_rcnn_postprocessing_resize` cannot work without "
            "metadata with `geometric_operations` field")
        coeff_x, coeff_y = self.get_coeff_x_y_from_metadata(image_metadata)

        for prediction in predictions:
            prediction.x_mins *= coeff_x
            prediction.x_maxs *= coeff_x
            prediction.y_mins *= coeff_y
            prediction.y_maxs *= coeff_y

        return annotations, predictions
