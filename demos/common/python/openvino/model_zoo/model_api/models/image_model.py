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

from .model import Model
from .utils import RESIZE_TYPES, pad_image, InputTransform


class ImageModel(Model):
    '''An abstract wrapper for image-based model

    An image-based model is model which has one or more inputs with image - 4D tensors with NWHC or NCHW layout.
    Also it may have support additional inputs - 2D tensor.
    Implements basic preprocessing for image: resizing and aligning to model input.

    Attributes:
        resize_type(str): one of the preimplemented resize types
        image_blob_names(List[str]): names of all image-like inputs (4D tensors)
        image_info_blob_names(List[str]): names of all secondary inputs (2D tensors)
        image_blob_name(str): name of image input (None, if they are many)
    '''

    def __init__(self, model_adapter, resize_type=None):
        '''Image model constructor

        Calls the `Model` constructor first

        Args:
            model_adapter(ModelAdapter): allows working with the specified executor
            resize_type(str): sets the type for image resizing (see ``RESIZE_TYPE`` for info)
        '''
        super().__init__(model_adapter)
        self.image_blob_names, self.image_info_blob_names = self._get_inputs()
        self.image_blob_name = self.image_blob_names[0] if len(self.image_blob_names) == 1 else None
        if self.image_blob_name:
            self.n, self.c, self.h, self.w = self.inputs[self.image_blob_name].shape

        self.image_layout = 'NCHW'
        if not resize_type:
            self.logger.warning('The resizer isn\'t set. The "standard" will be used')
            resize_type = 'standard'
        self.resize_type = resize_type
        self.resize = RESIZE_TYPES[self.resize_type]
        self.input_transform = InputTransform()

    def set_inputs_preprocessing(self, reverse_input_channels, mean_values, scale_values):
        self.input_transform = InputTransform(reverse_input_channels, mean_values, scale_values)

    def _get_inputs(self):
        image_blob_names, image_info_blob_names = [], []
        for name, metadata in self.inputs.items():
            if len(metadata.shape) == 4:
                image_blob_names.append(name)
            elif len(metadata.shape) == 2:
                image_info_blob_names.append(name)
            else:
                raise RuntimeError('Failed to identify the input for ImageModel: only 2D and 4D input layer supported')
        if not image_blob_names:
            raise RuntimeError('Failed to identify the input for the image: no 4D input layer found')
        return image_blob_names, image_info_blob_names

    def preprocess(self, inputs):
        '''Data preprocess method

        Performs some basic preprocessing with single image:
        - resizing to net input size
        - applying tranform orerations: mean and scale values, BGR-RGB conversions
        - changing layout according to net input layout

        Adds the size of initial image and after resizing to metadata as `original_shape` and `resized_shape`
        correspondenly.

        Note:
            This method supports only models with single image input. If model has more image inputs
            or has additional support inputs, their preprocessing should be implemented in concrete class

        Args:
            inputs: single image as 3D array in HWC layout

        Returns:
            - the dict with preprocessed image data
            - The dict with metadata
        '''
        image = inputs
        meta = {'original_shape': image.shape}
        resized_image = self.resize(image, (self.w, self.h))
        meta.update({'resized_shape': resized_image.shape})
        if self.resize_type == 'fit_to_window':
            resized_image = pad_image(resized_image, (self.w, self.h))
        resized_image = self.input_transform(resized_image)
        resized_image = self._change_layout(resized_image)
        dict_inputs = {self.image_blob_name: resized_image}
        return dict_inputs, meta

    def _change_layout(self, image):
        if self.image_layout == 'NCHW':
            image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            image = image.reshape((1, self.c, self.h, self.w))
        else:
            image = image.reshape((1, self.h, self.w, self.c))
        return image
