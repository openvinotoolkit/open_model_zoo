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
from .utils import resize_image, resize_image_with_aspect, resize_image_letterbox, pad_image


class ImageModel(Model):
    RESIZE_TYPES = {
        'default': resize_image,
        'keep_aspect_ratio': resize_image_with_aspect,
        'letterbox': resize_image_letterbox,
    }

    def __init__(self, ie, model_path, input_transform=None, resize_type='default'):
        super().__init__(ie, model_path, input_transform=input_transform)
        self.image_blob_name = self._get_image_input()
        if self.image_blob_name:
            self.n, self.c, self.h, self.w = self.net.input_info[self.image_blob_name].input_data.shape
        self.image_layout = 'CHW'
        self.resize_type = resize_type
        self.resize = self.RESIZE_TYPES[self.resize_type]

    def _get_image_input(self):
        image_blob_name = None
        for blob_name, blob in self.net.input_info.items():
            if len(blob.input_data.shape) == 4:
                if not image_blob_name:
                    image_blob_name = blob_name
                else:
                    raise RuntimeError('Failed to identify the input for image: more than one 4D input layer found')
        if image_blob_name is None:
            raise RuntimeError('Failed to identify the input for the image: no 4D input layer found')
        return image_blob_name

    def preprocess(self, inputs):
        image = inputs
        meta = {'original_shape': image.shape}


        # for image in inputs:
        #     resized_image, meta = self._preprocess_single_image(image)


        resized_image = self.resize(image, (self.w, self.h))
        meta.update({'resized_shape': resized_image.shape})
        if self.resize_type == 'keep_aspect_ratio':
            resized_image = pad_image(resized_image, (self.w, self.h))
        if self.input_transform:
            resized_image = self.input_transform(resized_image)
        resized_image = self._change_layout(resized_image)
        dict_inputs = {self.image_blob_name: resized_image}
        return dict_inputs, meta

    def _preprocess_single_image(self, image):
        meta = {'original_shape': image.shape}
        resized_image = self.resize(image, (self.w, self.h))
        resized_image = self.input_transform(resized_image)
        resized_image = self._change_layout(resized_image)
        meta.update({'resized_shape': resized_image.shape})
        dict_inputs = {self.image_blob_name: image}
        return resized_image, meta

    def _change_layout(self, image):
        if self.image_layout == 'CHW':
            image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            image = image.reshape((self.n, self.c, self.h, self.w))
        else:
            image = image.reshape((self.n, self.h, self.w, self.c))
        return image
