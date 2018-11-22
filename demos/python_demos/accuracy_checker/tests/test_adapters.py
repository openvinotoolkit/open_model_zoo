"""
 Copyright (c) 2018 Intel Corporation

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

from accuracy_checker.adapters import SSDAdapter
from .common import make_representation


def test_detection_adapter():
    raw = {
        'detection_out': np.array([[[[0, 3, 0.2, 0, 0, 1, 1], [0, 2, 0.5, 4, 4, 7, 7], [0, 5, 0.7, 3, 3, 9, 8]]]])
    }
    expected = make_representation('0.2,3,0,0,1,1;0.5,2,4,4,7,7;0.7,5,3,3,9,8')

    actual = SSDAdapter({}, output_blob='detection_out')(raw, ['0'])

    assert np.array_equal(actual, expected)


def test_detection_adapter_partially_filling_output_blob():
    raw = {
        'detection_out': np.array(
            [[[[0, 3, 0.2, 0, 0, 1, 1], [0, 2, 0.5, 4, 4, 7, 7], [0, 5, 0.7, 3, 3, 9, 8], [-1, 0, 0, 0, 0, 0, 0]]]]
        )
    }
    expected = make_representation('0.2,3,0,0,1,1;0.5,2,4,4,7,7;0.7,5,3,3,9,8')

    actual = SSDAdapter({}, output_blob='detection_out')(raw, ['0'])

    assert np.array_equal(actual, expected)

def test_detection_adapter_partially_filling_output_blob_with_zeros_at_the_end():
    raw = {
        'detection_out': np.array(
            [[[[0, 3, 0.2, 0, 0, 1, 1], [0, 2, 0.5, 4, 4, 7, 7], [0, 5, 0.7, 3, 3, 9, 8], [-1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]]]
        )
    }
    expected = make_representation('0.2,3,0,0,1,1;0.5,2,4,4,7,7;0.7,5,3,3,9,8')

    actual = SSDAdapter({}, output_blob='detection_out')(raw, ['0'])

    assert np.array_equal(actual, expected)


def test_detection_adapter_batch_2():
    raw = {
        'detection_out': np.array([[[[0, 3, 0.2, 0, 0, 1, 1], [0, 2, 0.5, 4, 4, 7, 7], [1, 5, 0.7, 3, 3, 9, 8]]]])
    }
    expected = make_representation(['0.2,3,0,0,1,1;0.5,2,4,4,7,7', '0.7,5,3,3,9,8'])

    actual = SSDAdapter({}, output_blob='detection_out')(raw, ['0', '1'])

    assert np.array_equal(actual, expected)
