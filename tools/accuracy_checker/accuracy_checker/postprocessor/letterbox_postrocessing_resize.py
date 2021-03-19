"""
Copyright (c) 2021 Intel Corporation
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


class LetterboxPostprocessingResize(Postprocessor):
    __provider__ = 'letterbox_postprocessing_resize'

    prediction_types = (DetectionPrediction, )
    annotation_types = (DetectionAnnotation, )

    def process_image(self, annotations, predictions):
        raise RuntimeError("Since `process_image_with_metadata` is overriden, this method MUST NOT be called")

    def process_image_with_metadata(self, annotations, predictions, image_metadata=None):
        assert image_metadata and 'geometric_operations' in image_metadata, (
            "Postprocessing step `faster_rcnn_postprocessing_resize` cannot work without "
            "metadata with `geometric_operations` field")

        assert image_metadata and 'geometric_operations' in image_metadata
        geometric_operations = image_metadata['geometric_operations']  # should be list of GeometricOperationMetadata

        if len(geometric_operations) == 2:
            op_name1, op_props1 = geometric_operations[0]
            op_name2, op_props2 = geometric_operations[1]
            assert op_name1 == "resize", (
                "Unknown case: two geometric operations, the first with name '{}'".format(op_name1))
            assert op_name2 == "padding", (
                "Unknown case: two geometric operations, the second with name '{}'".format(op_name2))
            pad = op_props2['pad']

            offset_x = pad[1] / op_props2['pref_width']
            offset_y = pad[2] / op_props2['pref_height']
            scale_x = op_props2['pref_width'] / op_props2['width'] * op_props1['original_width']
            scale_y = op_props2['pref_height'] / op_props2['height'] * op_props1['original_height']

        else:
            raise RuntimeError(
                "Unknown case: ""len(image_metadata['geometric_operations']) = {}".format(len(geometric_operations))
            )

        for prediction in predictions:
            prediction.x_mins = (prediction.x_mins - offset_x) * scale_x
            prediction.x_maxs = (prediction.x_maxs - offset_x) * scale_x
            prediction.y_mins = (prediction.y_mins - offset_y) * scale_y
            prediction.y_maxs = (prediction.y_maxs - offset_y) * scale_y

        return annotations, predictions
