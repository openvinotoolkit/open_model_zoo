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

import cv2
import numpy as np


def find_peaks_and_plateaus(heatmap, all_peaks, prev_peak_counter):
    """Prepare the heatmap, find peaks, and save the Accuracy Checker result.

    Accepts a non-empty 2D heatmap. Modifies it in place: sets NaNs
    and values below the existing 0.1 threshold to zero.
    For each accepted region, selects the actual pixel nearest its
    centroid. Then applies NMS, giving higher heatmap scores priority.

    Appends a list of [x, y, value, id] entries to all_peaks.
    Returns the number of points appended after NMS.
    """
    _prepare_heatmap(heatmap)
    coordinates = _find_peak_coordinates(heatmap)
    coordinates = _suppress_close_peaks(coordinates, heatmap)

    peaks = [
        [x, y, heatmap[y, x], prev_peak_counter + index]
        for index, (x, y) in enumerate(coordinates)
    ]
    all_peaks.append(peaks)
    return len(peaks)


def _prepare_heatmap(heatmap):
    """Set NaNs and responses below 0.1 to zero, modifying the heatmap in place.

    Purpose: preserve the existing decoder threshold. Background pixels
    and weak responses must not become candidates.
    """
    heatmap[heatmap < 0.1] = 0
    heatmap[np.isnan(heatmap)] = 0


def _find_peak_candidates(heatmap):
    """Step 1: mark positive pixels >= each of their four neighbors.

    Copies the heatmap into five separate arrays, as the original strict `>`
    decoder does. All arrays have the heatmap's dtype; out-of-image
    neighbors are treated as zero.
    Returns a bool mask with the heatmap's shape; True marks a candidate.
    A candidate is not necessarily a true maximum yet.
    """
    height, width = heatmap.shape
    center = np.zeros((height + 2, width + 2), dtype=heatmap.dtype)
    left = np.zeros_like(center)
    right = np.zeros_like(center)
    up = np.zeros_like(center)
    down = np.zeros_like(center)

    center[1:-1, 1:-1] = heatmap
    left[1:-1, :-2] = heatmap
    right[1:-1, 2:] = heatmap
    up[:-2, 1:-1] = heatmap
    down[2:, 1:-1] = heatmap

    candidates = center > 0
    for neighbor in (left, right, up, down):
        candidates &= center >= neighbor
    return candidates[1:-1, 1:-1]


def _find_peak_coordinates(heatmap):
    """Label and check each region individually, before NMS.

    OpenCV connects candidates that share an edge (connectivity=4).
    Label 0 is the background, so the loop starts at label 1.
    Check ALL regions, including single-pixel regions.

    Collect the coordinates of every pixel in a region.
    The stats bounding box limits the search, avoiding a full-image
    scan for each region.

    If any pixel meets the rejection condition, skip the entire region
    and continue to the next one. Otherwise, select the actual region
    pixel nearest its centroid. Break ties by smallest y, then x.

    Returns one (x, y) coordinate per accepted region, sorted by x and
    then y. Does not create suspected, rejected, or strict-peak masks.
    """
    candidates = _find_peak_candidates(heatmap)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        candidates.astype(np.uint8),
        connectivity=4,
    )

    coordinates = []
    for region_id in range(1, count):
        left, top, width, height = stats[region_id, :4]
        region = labels[top:top + height, left:left + width] == region_id
        ys, xs = np.nonzero(region)
        ys = ys + top
        xs = xs + left

        if _region_has_higher_pixel(heatmap, ys, xs):
            continue

        coordinates.append(_select_center_pixel(ys, xs, centroids[region_id]))

    return sorted(coordinates)


def _select_center_pixel(ys, xs, centroid):
    """Select the actual region pixel nearest its centroid.

    ys and xs are region pixel coordinates; centroid contains OpenCV's
    mean (x, y) position.
    For a single-pixel region, return its unchanged position.

    For a larger region, calculate squared distances from the centroid.
    Do not round the centroid itself: it could land outside the region,
    for example in the hole at the center of a ring.
    np.argmin selects the first equally close pixel: smallest y, then x.
    Returns integer (x, y) coordinates.
    """
    if len(xs) == 1:
        return int(xs[0]), int(ys[0])

    center_x, center_y = centroid
    distances_squared = (xs - center_x) ** 2 + (ys - center_y) ** 2
    nearest = np.argmin(distances_squared)
    return int(xs[nearest]), int(ys[nearest])


def _region_has_higher_pixel(heatmap, ys, xs):
    """Step 2: visit region pixels and check four cardinal directions.

    ys and xs are the coordinates of every pixel in one region.
    For each pixel P, check only this pattern:

        P -> neighbor -> farther pixel
        2      2           5

    The neighbor must equal P, and the farther pixel must exceed P.
    Read from the ORIGINAL heatmap, not the candidate mask.
    A neighbor's value is still present even if it is not a candidate.

    2D example (2 and 5 are illustrative values):

        heatmap:        labels from the >= mask:
        0 0 0 0 0 0     0 0 0 0 0 0
        0 2 2 2 5 0     0 1 1 0 2 0
        0 0 0 0 0 0     0 0 0 0 0 0

    The first pixel of region 1 does not detect a higher value,
    but the second sees 2 to its right, then 5. Return True:
    stop checking this region and reject it entirely.

    Limitation: turns are not checked. Here, the region of lower
    candidates with value 2 remains accepted despite the nearby 5:

        0 0 0 5 0
        0 2 2 2 0
        0 0 0 0 0

    The path from the last candidate to 5 goes right, then up.
    It is not a straight line, so it does not meet the requested rule.

    Return True on the first match, False after checking the region.
    Directions are up, down, left, and right. No diagonals or epsilon.
    """
    height, width = heatmap.shape
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))

    for y, x in zip(ys, xs):
        value = heatmap[y, x]
        for dy, dx in directions:
            farther_y = y + 2 * dy
            farther_x = x + 2 * dx
            if not (0 <= farther_y < height and 0 <= farther_x < width):
                continue

            neighbor_y = y + dy
            neighbor_x = x + dx
            if heatmap[neighbor_y, neighbor_x] != value:
                continue

            if heatmap[farther_y, farther_x] > value:
                return True

    return False


def _suppress_close_peaks(coordinates, heatmap):
    """Suppress nearby peaks in descending heatmap score order.

    Equal scores retain the input (x, y) order. A point less than 6
    pixels from an accepted peak is rejected. Return the survivors in
    (x, y) order to preserve peak numbering order for pose grouping.
    """
    accepted = []
    for x, y in sorted(coordinates, key=lambda point: heatmap[point[1], point[0]], reverse=True):
        if any((x - px) ** 2 + (y - py) ** 2 < 36 for px, py in accepted):
            continue
        accepted.append((x, y))
    return sorted(accepted)
