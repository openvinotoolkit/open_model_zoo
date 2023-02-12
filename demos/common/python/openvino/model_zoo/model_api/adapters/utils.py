"""
 Copyright (C) 2022-2023 Intel Corporation

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
import openvino.runtime as ov

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
    
    if desired_aspect_ratio == 1:
        # then_body
        image_t = opset.parameter([-1,-1,-1,3], np.uint8, "image")
        iw_t = opset.parameter([], np.int32, "iw")
        ih_t = opset.parameter([], np.int32, "ih")
        then_offset = opset.unsqueeze(opset.divide(opset.subtract(ih_t, iw_t), opset.constant(2, dtype=np.int32)), 0)
        then_stop = opset.add(then_offset, iw_t)
        then_cropped_frame = opset.slice(image_t, start=then_offset, stop=then_stop, step=[1], axes=[h_axis])
        then_body_res_1 = opset.result(then_cropped_frame)
        then_body = ov.Model([then_body_res_1], [image_t, iw_t, ih_t], "then_body_function")

        # else_body
        image_e = opset.parameter([-1,-1,-1,3], np.uint8, "image")
        iw_e = opset.parameter([], np.int32, "iw")
        ih_e = opset.parameter([], np.int32, "ih")
        else_offset = opset.unsqueeze(opset.divide(opset.subtract(iw_e, ih_e), opset.constant(2, dtype=np.int32)), 0)
        else_stop = opset.add(else_offset, ih_e)
        else_cropped_frame = opset.slice(image_e, start=else_offset, stop=else_stop, step=[1], axes=[w_axis])
        else_body_res_1 = opset.result(else_cropped_frame)
        else_body = ov.Model([else_body_res_1], [image_e, iw_e, ih_e], "else_body_function")

        # if
        condition = opset.greater(ih, iw)
        if_node = opset.if_op(condition.output(0))
        if_node.set_then_body(then_body)
        if_node.set_else_body(else_body)
        if_node.set_input(input, image_t, image_e)
        if_node.set_input(iw.output(0), iw_t, iw_e)
        if_node.set_input(ih.output(0), ih_t, ih_e)
        cropped_frame = if_node.set_output(then_body_res_1, else_body_res_1)
        
    elif desired_aspect_ratio < 1:
        new_width = opset.floor(opset.multiply(opset.convert(ih, destination_type="f32"), desired_aspect_ratio))
        offset = opset.unsqueeze(opset.divide(opset.subtract(iw, new_width), opset.constant(2, dtype=np.int32)), 0)
        stop = opset.add(offset, new_width)
        cropped_frame = opset.slice(input, start=offset, stop=stop, step=[1], axes=[w_axis])
    elif desired_aspect_ratio > 1:
        new_hight = opset.floor(opset.multiply(opset.convert(iw, destination_type="f32"), desired_aspect_ratio))
        offset = opset.unsqueeze(opset.divide(opset.subtract(ih, new_hight), opset.constant(2, dtype=np.int32)), 0)
        stop = opset.add(offset, new_hight)
        cropped_frame = opset.slice(input, start=offset, stop=stop, step=[1], axes=[h_axis])
    
    target_size = list(size)
    target_size.reverse()
    resized_image = opset.interpolate(cropped_frame, target_size, scales=np.array([1.0, 1.0], dtype=np.float32),
                              axes=[h_axis, w_axis], 
                              mode="linear", shape_calculation_mode="sizes")
    return resized_image


def resize_image_graph(input: Output, size, keep_aspect_ratio=False, interpolation="linear"):
    h_axis = 1
    w_axis = 2
    w, h = size
    
    target_size = list(size)
    target_size.reverse()
    
    if not keep_aspect_ratio:
        resized_image = opset.interpolate(input, target_size, scales=np.array([1.0, 1.0], dtype=np.float32),
                              axes=[h_axis, w_axis], 
                              mode="linear", shape_calculation_mode="sizes")
    else:
        image_shape = opset.shape_of(input, name="shape")
        iw = opset.convert(opset.gather(image_shape, opset.constant(w_axis), axis=0), destination_type="f32")
        ih = opset.convert(opset.gather(image_shape, opset.constant(h_axis), axis=0), destination_type="f32")
        w_ratio = opset.divide(np.float32(w), iw)
        h_ratio = opset.divide(np.float32(h), ih)
        scale = opset.minimum(w_ratio, h_ratio)
        resized_image = opset.interpolate(input, target_size, scales=scale,
                              axes=[h_axis, w_axis], 
                              mode="linear", shape_calculation_mode="sizes")
    return resized_image


def resize_image(size, interpolation="linear"):
    return custom_preprocess_function(partial(resize_image_graph, size=size, interpolation=interpolation))


def resize_image_with_aspect(size, interpolation="linear"):
   return custom_preprocess_function(partial(resize_image_graph, size=size, keep_aspect_ratio=True, interpolation=interpolation))


def crop_resize(size, interpolation="linear"):
    return custom_preprocess_function(partial(crop_resize_graph, size=size))


def resize_image_letterbox(size, interpolation="linear"):
    return custom_preprocess_function(partial(resize_image_letterbox_graph, size=size, interpolation=interpolation))
    