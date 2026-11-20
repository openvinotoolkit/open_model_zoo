"""
Copyright (c) 2026 Intel Corporation

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

from accuracy_checker.adapters.region_peak_decoder import find_peaks_and_plateaus


def test_keeps_one_peak_for_two_equal_neighbors():
    heatmap = np.array([[0, 0.5, 0.5, 0]], dtype=np.float32)
    all_peaks = []

    assert find_peaks_and_plateaus(heatmap, all_peaks, 7) == 1
    assert all_peaks == [[[1, 0, np.float32(0.5), 7]]]


def test_selects_region_pixel_nearest_centroid_when_centroid_is_in_a_hole():
    heatmap = np.zeros((5, 5), dtype=np.float32)
    heatmap[1:4, 1:4] = 0.5
    heatmap[2, 2] = 0
    all_peaks = []

    assert find_peaks_and_plateaus(heatmap, all_peaks, 0) == 1
    assert all_peaks == [[[2, 1, np.float32(0.5), 0]]]


def test_rejects_flat_shoulder_next_to_higher_peak():
    heatmap = np.zeros((3, 12), dtype=np.float32)
    heatmap[1, 1:10] = 0.25
    heatmap[1, 10] = 0.5
    all_peaks = []

    assert find_peaks_and_plateaus(heatmap, all_peaks, 0) == 1
    assert all_peaks == [[[10, 1, np.float32(0.5), 0]]]
