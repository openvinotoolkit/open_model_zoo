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

import numpy as np
import sys
import cv2
from PIL import Image


def postprocess(out):
    out = np.squeeze(out)
    out = np.transpose(out, (1, 2, 0))
    result = np.uint8(127.5 * out + 127.5)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result


def save_result(result, out_path):
    try:
        result = postprocess(result)
        cv2.imwrite(out_path, result)
    except OSError as err:
        print(err)
