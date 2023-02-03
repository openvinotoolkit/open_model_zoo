"""
 Copyright (C) 2022 Intel Corporation

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

from typing import Optional
from functools import partial

import numpy as np
from openvino.runtime import layout_helpers
from openvino.runtime import opset10 as opset
from openvino.runtime.utils.decorators import custom_preprocess_function
from openvino.runtime import Output

class Layout:
    def __init__(self, layout = '') -> None:
        self.layout = layout

    @staticmethod
    def from_shape(shape):
        '''
        Create Layout from given shape
        '''
        if len(shape) == 2:
            return 'NC'
        if len(shape) == 3:
            return 'CHW' if shape[0] in range(1, 5) else 'HWC'
        if len(shape) == 4:
            return 'NCHW' if shape[1] in range(1, 5) else 'NHWC'

        raise RuntimeError("Get layout from shape method doesn't support {}D shape".format(len(shape)))

    @staticmethod
    def from_openvino(input):
        '''
        Create Layout from openvino input
        '''
        return layout_helpers.get_layout(input).to_string().strip('[]').replace(',', '')

    @staticmethod
    def from_user_layouts(input_names: set, user_layouts: dict):
        '''
        Create Layout for input based on user info
        '''
        for input_name in input_names:
            if input_name in user_layouts:
                return user_layouts[input_name]
        return user_layouts.get('', '')

    @staticmethod
    def parse_layouts(layout_string: str) -> Optional[dict]:
        '''
        Parse layout parameter in format "input0:NCHW,input1:NC" or "NCHW" (applied to all inputs)
        '''
        if not layout_string:
            return None
        search_string = layout_string if layout_string.rfind(':') != -1 else ":" + layout_string
        colon_pos = search_string.rfind(':')
        user_layouts = {}
        while (colon_pos != -1):
            start_pos = search_string.rfind(',')
            input_name = search_string[start_pos + 1:colon_pos]
            input_layout = search_string[colon_pos + 1:]
            user_layouts[input_name] = input_layout
            search_string = search_string[:start_pos + 1]
            if search_string == "" or search_string[-1] != ',':
                break
            search_string = search_string[:-1]
            colon_pos = search_string.rfind(':')
        if search_string != "":
            raise ValueError("Can't parse input layout string: " + layout_string)
        return user_layouts
    

def resize_image_letterbox_graph(input: Output, size, interpolation="linear"):
    w, h = size
    h_axis = 1
    w_axis = 2
    image_shape = opset.shape_of(input, name="shape")
    iw = opset.convert(opset.gather(image_shape, opset.constant(w_axis), axis=0), destination_type="f32")
    ih = opset.convert(opset.gather(image_shape, opset.constant(h_axis), axis=0), destination_type="f32")
    w_ratio = opset.divide(opset.constant(w, dtype=float), iw)
    h_ratio = opset.divide(opset.constant(h, dtype=float), ih)
    scale = opset.minimum(w_ratio, h_ratio)
    nw = opset.convert(opset.multiply(iw, scale), destination_type="i32")
    nh = opset.convert(opset.multiply(ih, scale), destination_type="i32")
    new_size = opset.concat([opset.unsqueeze(nh, 0) , opset.unsqueeze(nw, 0)], axis=-1)
    image = opset.interpolate(input, new_size, scales=np.array([1.0, 1.0], dtype=np.float32),
                              axes=[h_axis, w_axis], 
                              mode=interpolation, shape_calculation_mode="sizes")
    dx = opset.divide(opset.subtract(opset.constant(w, dtype=np.int32), nw), opset.constant(2, dtype=np.int32))
    dy = opset.divide(opset.subtract(opset.constant(h, dtype=np.int32), nh), opset.constant(2, dtype=np.int32))
    dx_border = opset.add(dx, opset.mod(opset.subtract(opset.constant(w, dtype=np.int32), nw), opset.constant(2, dtype=np.int32)))
    dy_border = opset.add(dy, opset.mod(opset.subtract(opset.constant(h, dtype=np.int32), nh), opset.constant(2, dtype=np.int32)))
    pads_begin = opset.concat([opset.constant([0], dtype=np.int32), 
                               opset.unsqueeze(dy, 0),
                               opset.unsqueeze(dx, 0), 
                               opset.constant([0], dtype=np.int32)], 
                              axis=0)
    pads_end = opset.concat([opset.constant([0], dtype=np.int32), 
                            opset.unsqueeze(dy_border, 0),
                            opset.unsqueeze(dx_border, 0), 
                            opset.constant([0], dtype=np.int32)], axis=0)
    resized_image = opset.pad(image, pads_begin, pads_end, "constant")
    return resized_image

def crop_resize_graph(input: Output, size):
    h_axis = 1
    w_axis = 2
    desired_aspect_ratio = size[1] / size[0] # width / height
    image_shape = opset.shape_of(input, name="shape")
    
    iw = opset.convert(opset.gather(image_shape, opset.constant(w_axis), axis=0), destination_type="i32")
    ih = opset.convert(opset.gather(image_shape, opset.constant(h_axis), axis=0), destination_type="i32")
    
    #if desired_aspect_ratio == 1:
    offset = opset.unsqueeze(opset.divide(opset.subtract(ih, iw), opset.constant(2, dtype=np.int32)), 0)
    stop = opset.add(offset, iw)
    cropped_frame = opset.slice(input, start=offset, stop=stop, step=[1], axes=[h_axis])
    
    resized_image = opset.interpolate(cropped_frame, list(size), scales=np.array([1.0, 1.0], dtype=np.float32),
                              axes=[h_axis, w_axis], 
                              mode="linear", shape_calculation_mode="sizes")
    return resized_image


# def crop_resize(image, size):
#     desired_aspect_ratio = size[1] / size[0] # width / height
#     if desired_aspect_ratio == 1:
#         if (image.shape[0] > image.shape[1]):
#             offset = (image.shape[0] - image.shape[1]) // 2
#             cropped_frame = image[offset:image.shape[1] + offset]
#         else:
#             offset = (image.shape[1] - image.shape[0]) // 2
#             cropped_frame = image[:, offset:image.shape[0] + offset]
#     elif desired_aspect_ratio < 1:
#         new_width = math.floor(image.shape[0] * desired_aspect_ratio)
#         offset = (image.shape[1] - new_width) // 2
#         cropped_frame = image[:, offset:new_width + offset]
#     elif desired_aspect_ratio > 1:
#         new_height = math.floor(image.shape[1] / desired_aspect_ratio)
#         offset = (image.shape[0] - new_height) // 2
#         cropped_frame = image[offset:new_height + offset]

#     return cv2.resize(cropped_frame, size)

def crop_resize(size, interpolation="linear"):
    return custom_preprocess_function(partial(crop_resize_graph, size=size))

def resize_image_letterbox(size, interpolation="linear"):
    return custom_preprocess_function(partial(resize_image_letterbox_graph, size=size, interpolation=interpolation))
    
    
# def resize_image_letterbox(image, size, interpolation=cv2.INTER_LINEAR):
#     ih, iw = image.shape[0:2]
#     w, h = size
#     scale = min(w / iw, h / ih)
#     nw = int(iw * scale)
#     nh = int(ih * scale)
#     image = cv2.resize(image, (nw, nh), interpolation=interpolation)
#     dx = (w - nw) // 2
#     dy = (h - nh) // 2
#     resized_image = np.pad(image, ((dy, dy + (h - nh) % 2), (dx, dx + (w - nw) % 2), (0, 0)),
#                            mode='constant', constant_values=0)
#     return resized_image