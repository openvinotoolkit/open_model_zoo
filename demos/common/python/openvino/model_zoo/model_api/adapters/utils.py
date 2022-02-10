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

import re
from typing import Optional
from openvino.runtime import layout_helpers


class Layout:
    def __init__(self, layout = '') -> None:
        self.layout = layout

    @classmethod
    def from_shape(cls, shape):
        '''
        Create Layout from given shape
        '''
        if len(shape) != 4:
            raise RuntimeError('Get layout from shape method supports only 4D input shape')

        layout = 'NCHW' if shape[1] in range(1, 4) else 'NHWC'
        return cls(layout)

    @classmethod
    def from_openvino(cls, input):
        '''
        Create Layout from openvino input
        '''
        return cls(layout_helpers.get_layout(input).to_string().strip('[]').replace(',', ''))

    @classmethod
    def from_user_layouts(cls, input_name, user_layouts: dict):
        '''
        Create Layout from user input
        '''
        if input_name in user_layouts:
            return cls(user_layouts[input_name])
        if '' in user_layouts:
            return cls(user_layouts[''])
        return cls()


    @staticmethod
    def parse_layouts(layout_regex: str) -> Optional[dict]:
        '''
        Parse layout parameter in format "input0:NCHW,input1:NC" or "NCHW" (applied to all inputs)
        '''
        if not layout_regex:
            return None
        inputs_layouts = layout_regex.split(',')
        user_layouts = {}
        for layout in inputs_layouts:
            layout_list = layout.split(':')
            if len(layout_list) == 2:
                input_name, input_layout = layout_list
            else:
                input_name = ''
                input_layout = layout_list[0]
            if re.fullmatch(r"[A-Z]+", input_layout):
                user_layouts[input_name] = input_layout
            else:
                raise ValueError("invalid --layout option format")
        return user_layouts
