"""
 Copyright (C) 2020 Intel Corporation

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

def put_highlighted_text(frame, message, position, font_face, font_scale, color, thickness):
    cv2.putText(frame, message, position, font_face, font_scale, (255, 255, 255), thickness + 1) # white border
    cv2.putText(frame, message, position, font_face, font_scale, color, thickness)

def resolution(value):
    try:
        result = [int(v) for v in value.split('x')]
        if len(result) != 2:
            raise RuntimeError('Сorrect format of --output_resolution parameter is "width"x"height".')
    except ValueError:
        raise RuntimeError('Сorrect format of --output_resolution parameter is "width"x"height".')
    return result
