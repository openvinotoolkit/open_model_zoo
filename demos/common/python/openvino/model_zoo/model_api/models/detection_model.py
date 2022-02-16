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
from .types import ListValue, NumericalValue, StringValue
from .model import WrapperError
from .image_model import ImageModel
from .utils import load_labels, clip_detections


class DetectionModel(ImageModel):
    '''An abstract detection model class

    This class supports detection models. The Detection Model must have single image input.

    Attributes:
        labels(List[str]): list of labels for classes (could be None)
        threshold(float): threshold for detection filtering, any detection with confidence less than this value
            should be omitted in ``posptrocess`` method (0<=thresold<=1.0 for most models)
        iou_threshold(float): threshold for NMS detection filtering
    '''

    def __init__(self, model_adapter, configuration=None, preload=False):
        '''The Detection Model constructor

        Calls the ``ImageModel`` construtor first.

        Args:
            labels(Iterable[str], str, Path): list of labels for detection classes or path to file with them
            threshold(float): threshold for detections filtering by confidence
            iou_threshold(float): threshold for NMS filtering

        Raises:
            RuntimeError: If loaded model has more than one image inputs
        '''
        super().__init__(model_adapter, configuration, preload)

        if not self.image_blob_name:
            raise WrapperError(self.__model__, "The Wrapper supports only one image input, but {} found"
                               .format(len(self.image_blob_names)))

        if self.path_to_labels:
            self.labels = load_labels(self.path_to_labels)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'confidence_threshold': NumericalValue(default_value=0.5, description="Threshold value for detection box confidence"),
            'labels': ListValue(description="List of class labels"),
            'path_to_labels': StringValue(
                description="Path to file with labels. Overrides the labels, if they sets via 'labels' parameter"
            )
        })

        return parameters

    def _resize_detections(self, detections, meta):
        '''Resizes detection bounding boxes according to initial image size

        Implements resize operations for different image resize types (see ``ImageModel`` class for details).
        Applies clipping bounding box to original image size.

        Args:
            detections(List[Detection]): list of detections with coordinates in normalized form
            meta: meta information with fields `resized_shape` and `original_shape`

        Returns:
            List of detections fit to initial image (resized and clipped)

        Raises:
            RuntimeError: If model uses custom resize or `resize_type` not set
        '''
        resized_shape = meta['resized_shape']
        original_shape = meta['original_shape']

        if self.resize_type=='fit_to_window_letterbox':
            detections = resize_detections_letterbox(detections, original_shape[1::-1], resized_shape[1::-1])
        elif self.resize_type == 'fit_to_window':
            detections = resize_detections_with_aspect_ratio(detections, original_shape[1::-1], resized_shape[1::-1], (self.w, self.h))
        elif self.resize_type == 'standard':
            detections = resize_detections(detections, original_shape[1::-1])
        else:
            raise WrapperError(self.__model__, 'Unknown resize type {}'.format(self.resize_type))
        return clip_detections(detections, original_shape)


def resize_detections(detections, original_image_size):
    for detection in detections:
        detection.xmin *= original_image_size[0]
        detection.xmax *= original_image_size[0]
        detection.ymin *= original_image_size[1]
        detection.ymax *= original_image_size[1]
    return detections

def resize_detections_with_aspect_ratio(detections, original_image_size, resized_image_size, model_input_size):
    scale_x = model_input_size[0] / resized_image_size[0] * original_image_size[0]
    scale_y = model_input_size[1] / resized_image_size[1] * original_image_size[1]
    for detection in detections:
        detection.xmin *= scale_x
        detection.xmax *= scale_x
        detection.ymin *= scale_y
        detection.ymax *= scale_y
    return detections

def resize_detections_letterbox(detections, original_image_size, resized_image_size):
    scales = [x / y for x, y in zip(resized_image_size, original_image_size)]
    scale = min(scales)
    scales = (scale / scales[0], scale / scales[1])
    offset = [0.5 * (1 - x) for x in scales]
    for detection in detections:
        detection.xmin = ((detection.xmin - offset[0]) / scales[0]) * original_image_size[0]
        detection.xmax = ((detection.xmax - offset[0]) / scales[0]) * original_image_size[0]
        detection.ymin = ((detection.ymin - offset[1]) / scales[1]) * original_image_size[1]
        detection.ymax = ((detection.ymax - offset[1]) / scales[1]) * original_image_size[1]
    return detections
