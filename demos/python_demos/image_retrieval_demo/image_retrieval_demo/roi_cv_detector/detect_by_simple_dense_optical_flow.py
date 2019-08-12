# Copyright (C) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

#pylint: disable=C0103,W1202,C0302
"""
The module contains the class RoiDetector that allows to detect ROI on frames received
from a video sequence using motion detection + background subtraction approaches.
"""

from collections import namedtuple
import math
import datetime
import logging as log
import cv2
import numpy as np

# stores rectangle as tuple of left_x, top_y, width, and height
Rect = namedtuple("Rect", ["tl_x", "tl_y", "w", "h"])
Point = namedtuple("Point", ["x", "y"])

IncreaseRectParams = namedtuple("IncreaseRectParams", ["increase_cell_coeff", "shift_x", "shift_y"])
CellParams = namedtuple("CellParams", ["cell_height", "cell_aspect_ratio", "cell_overlap",
                                       "num_cells_x", "num_cells_y", "list_v_len"])

CellData = namedtuple("CellData", ["rect", "increased_rect", "list_v", "calculated"])

ConnectedComponent = namedtuple("ConnectedComponent",
                                ["label_id", "mask", "centroid", "rect", "area", "num"])


SHOULD_SHOW = False
def _my_imshow(name, img):
    if SHOULD_SHOW:
        cv2.imshow(name, img)


SHOULD_SHOW_DEBUG = False
def _dbg_imshow(name, img):
    if SHOULD_SHOW_DEBUG:
        _my_imshow(name, img)


def _log_work_time(prefix, name_point, begin_work_time):
    work_dt = datetime.datetime.now() - begin_work_time
    work_dt_ms = int(1000*work_dt.total_seconds())
    log.debug("{}: {} = {} ms".format(prefix, name_point, work_dt_ms))


def _get_rect_in_center(image_shape, size):
    H, W = image_shape[:2]
    w, h = size
    tl_x = int(W / 2. - w / 2.)
    tl_y = int(H / 2. - h / 2.)
    return Rect(tl_x, tl_y, w, h)


def _get_center_fp(rect):
    tl_x, tl_y, w, h = rect
    c_x = tl_x + w/2.
    c_y = tl_y + h/2.
    return Point(c_x, c_y)


def _get_center(rect):
    c_x, c_y = _get_center_fp(rect)
    c_x = int(c_x)
    c_y = int(c_y)
    return Point(c_x, c_y)


def _increase_rect(rect, increase_rect_params):
    coeff = increase_rect_params.increase_cell_coeff
    shift_x = increase_rect_params.shift_x
    shift_y = increase_rect_params.shift_y

    _, _, w, h = rect

    new_w = math.ceil(w * coeff + 2*shift_x)
    new_h = math.ceil(h * coeff + 2*shift_y)

    c_x, c_y = _get_center_fp(rect)
    new_tl_x = math.floor(c_x - new_w / 2.)
    new_tl_y = math.floor(c_y - new_h / 2.)

    return Rect(new_tl_x, new_tl_y, new_w, new_h)


def get_rect_tl(rect):
    """
    Returns namedtuple Point that is top-left corner of the given
    namedtuple Rect
    """
    return Point(rect[0], rect[1])


def get_rect_br(rect):
    """
    Returns namedtuple Point that is bottom-right corner of the given
    namedtuple Rect
    """
    tl_x, tl_y, w, h = rect
    return Point(tl_x + w - 1, tl_y + h - 1)


#pylint: disable=R0914
def _intersect_rects(rect1, rect2):
    if rect1 is None or rect2 is None:
        return None
    tl_x_1, tl_y_1 = get_rect_tl(rect1)
    tl_x_2, tl_y_2 = get_rect_tl(rect2)
    tl_x = max(tl_x_1, tl_x_2)
    tl_y = max(tl_y_1, tl_y_2)

    br_x_1, br_y_1 = get_rect_br(rect1)
    br_x_2, br_y_2 = get_rect_br(rect2)
    br_x = min(br_x_1, br_x_2)
    br_y = min(br_y_1, br_y_2)

    if br_x < tl_x or br_y < tl_y:
        return Rect(0, 0, 0, 0)

    w = br_x - tl_x + 1
    h = br_y - tl_y + 1

    return Rect(tl_x, tl_y, w, h)

#pylint: disable=R0914
def _get_union_rects(rect1, rect2):
    if rect1 is None or rect2 is None:
        return None
    tl_x_1, tl_y_1 = get_rect_tl(rect1)
    tl_x_2, tl_y_2 = get_rect_tl(rect2)
    tl_x = min(tl_x_1, tl_x_2)
    tl_y = min(tl_y_1, tl_y_2)

    br_x_1, br_y_1 = get_rect_br(rect1)
    br_x_2, br_y_2 = get_rect_br(rect2)
    br_x = max(br_x_1, br_x_2)
    br_y = max(br_y_1, br_y_2)

    if br_x < tl_x or br_y < tl_y:
        return Rect(0, 0, 0, 0)

    w = br_x - tl_x + 1
    h = br_y - tl_y + 1

    return Rect(tl_x, tl_y, w, h)


def _get_area_rect(rect):
    if rect is None:
        return None
    assert rect.w >= 0 and rect.h >= 0
    return rect.w * rect.h


def _get_iou_rects(rect1, rect2):
    if rect1 is None or rect2 is None:
        return None
    rect12 = _intersect_rects(rect1, rect2)
    a1 = _get_area_rect(rect1)
    a2 = _get_area_rect(rect2)
    a12 = _get_area_rect(rect12)
    return float(a12) / (a1 + a2 - a12)


def _scale_rect(rect, scale):
    scaled_vals = [int(scale * v) for v in rect]
    return Rect(*scaled_vals)


def _get_subimage(image, rect):
    tl_x, tl_y, w, h = rect
    subimage = image[tl_y : tl_y + h, tl_x : tl_x + w, :]

    assert subimage.shape[0] == h
    assert subimage.shape[1] == w

    return subimage.copy()


def _get_median_of_rects(rects):
    list_tl_x = []
    list_tl_y = []
    list_br_x = []
    list_br_y = []
    for r in rects:
        tl_x, tl_y = get_rect_tl(r)
        br_x, br_y = get_rect_br(r)
        list_tl_x.append(tl_x)
        list_tl_y.append(tl_y)
        list_br_x.append(br_x)
        list_br_y.append(br_y)

    list_tl_x = np.array(list_tl_x)
    list_tl_y = np.array(list_tl_y)
    list_br_x = np.array(list_br_x)
    list_br_y = np.array(list_br_y)

    tl_x = np.median(list_tl_x)
    tl_y = np.median(list_tl_y)
    br_x = np.median(list_br_x)
    br_y = np.median(list_br_y)

    w = br_x - tl_x + 1
    h = br_y - tl_y + 1

    return Rect(tl_x, tl_y, w, h)


def _draw_match(match, min_val, max_val, show_size_coeff=10):
    if not SHOULD_SHOW:
        return
    divisor = (max_val - min_val) if max_val > min_val else 1.
    match_to_draw = (match - min_val) / divisor
    h, w = match_to_draw.shape[:2]
    match_to_draw = cv2.resize(match_to_draw, (show_size_coeff * w, show_size_coeff * h))
    _dbg_imshow("match", match_to_draw)


def _run_match_template_on_rect(image, prev_image, rect, increased_rect):
    subimage = _get_subimage(image, increased_rect)
    prev_template = _get_subimage(prev_image, rect)

    match = cv2.matchTemplate(subimage, prev_template, cv2.TM_SQDIFF)

    min_val, max_val, min_loc, _ = cv2.minMaxLoc(match)

    dx, dy = min_loc
    template_h, template_w = prev_template.shape[:2]
    subimage_h, subimage_w = subimage.shape[:2]

    v_x = -(subimage_w / 2.) + dx + template_w / 2.
    v_y = -(subimage_h / 2.) + dy + template_h / 2.

    v = Point(v_x, v_y)

    _draw_match(match, min_val, max_val)

    return v


def _draw_arrow(image, pt, v, color1=(0, 255, 0), color2=None):
    if not SHOULD_SHOW:
        return
    if color2 is None:
        color2 = color1
    pt = np.array(pt)
    v = np.array(v)
    pt2_a = pt + v
    pt2_b = pt + (v / 2.)

    pt2_a = pt2_a.astype(np.int32)
    pt2_b = pt2_b.astype(np.int32)
    cv2.line(image, tuple(pt), tuple(pt2_b), color=color2, thickness=3)
    cv2.line(image, tuple(pt), tuple(pt2_a), color=color1, thickness=1)


def _draw_rect(image, rect, color=(255, 0, 0), thickness=3):
    if not SHOULD_SHOW:
        return
    if rect is None:
        return
    tl_x, tl_y, w, h = rect
    br_x = tl_x + w
    br_y = tl_y + h

    cv2.rectangle(image, (tl_x, tl_y), (br_x, br_y), color, thickness)


def _decrease_image_to_min_side(image, target_min_side):
    h, w = image.shape[:2]

    min_side = min(h, w)
    scale = float(target_min_side) / min_side

    new_w = math.ceil(scale * w)
    new_h = math.ceil(scale * h)

    image = cv2.resize(image, (new_w, new_h))
    return image


def _get_median_from_list(list_v, N_median):
    xs = np.array([x for x, y in list_v[-N_median:]])
    median_x = np.median(xs)

    ys = np.array([y for x, y in list_v[-N_median:]])
    median_y = np.median(ys)

    return Point(median_x, median_y)


def _check_is_rect_valid(rect, frame_shape):
    tl_x, tl_y, w, h = rect
    frame_h, frame_w = frame_shape[:2]
    if tl_x < 0 or tl_y < 0:
        return False

    br_x = tl_x + w - 1
    br_y = tl_y + h - 1
    if br_x > frame_w or br_y > frame_h:
        return False
    return True


#pylint: disable=R0903
class RoiMotionDetector:
    """
    The class estimates regular motion in central region of a frame.
    """
    @staticmethod
    #pylint: disable=R0914
    def _init_cell_data(i, j, frame_shape, cell_params, increase_rect_params):
        frame_h, frame_w = frame_shape[:2]
        frame_cx = frame_w / 2.
        frame_cy = frame_h / 2.
        cell_h = cell_params.cell_height
        cell_w = int(cell_params.cell_aspect_ratio * cell_h)

        assert cell_params.num_cells_x % 2 == 1 and cell_params.num_cells_y % 2 == 1
        rel_i = i - cell_params.num_cells_y // 2
        rel_j = j - cell_params.num_cells_x // 2

        cell_cx = frame_cx + rel_j * cell_w
        cell_cy = frame_cy + rel_i * cell_h
        tl_x = math.floor(cell_cx - cell_w / 2.)
        tl_y = math.floor(cell_cy - cell_h / 2.)
        cell_rect = Rect(tl_x, tl_y, cell_w, cell_h)
        cell_increased_rect = _increase_rect(cell_rect, increase_rect_params)

        is_valid = _check_is_rect_valid(cell_rect, frame_shape)
        is_increased_valid = _check_is_rect_valid(cell_increased_rect, frame_shape)
        log.debug("_init_cell_data: (i,j) = {}, (rel_i, rel_j) = {}, cell_rect = {}, "
                  "cell_increased_rect = {}, is_valid={}, is_increased_valid={}".format(
                      (i, j), (rel_i, rel_j), cell_rect, cell_increased_rect,
                      is_valid, is_increased_valid))

        if not is_valid or not is_increased_valid:
            return None

        cell_data = CellData(rect=cell_rect, increased_rect=cell_increased_rect,
                             list_v=[], calculated={"median": None})
        return cell_data

    #pylint: disable=R0913
    def __init__(self, frame_shape, cell_params, increase_rect_params, N_median, min_motion=6):
        self.frame_shape = frame_shape
        self.cell_params = cell_params
        self.increase_rect_params = increase_rect_params
        self.N_median = N_median

        self.min_motion = min_motion
        self.total_v = Point(0, 0)

        self.cells_data = {}
        for i in range(self.cell_params.num_cells_y):
            for j in range(self.cell_params.num_cells_x):
                cell_data = self._init_cell_data(i, j, frame_shape=frame_shape,
                                                 cell_params=cell_params,
                                                 increase_rect_params=increase_rect_params)
                if cell_data is None:
                    continue
                self.cells_data[(i, j)] = cell_data

    def _handle_cell(self, cell_data, frame, prev_frame):
        rect = cell_data.rect
        increased_rect = cell_data.increased_rect

        assert _check_is_rect_valid(rect, frame.shape)
        assert _check_is_rect_valid(increased_rect, frame.shape)

        v = _run_match_template_on_rect(frame, prev_frame, rect, increased_rect)
        log.debug("    v = {}".format(v))

        cell_data.list_v.append(v)
        while len(cell_data.list_v) > self.cell_params.list_v_len:
            del cell_data.list_v[0]

    def _recalculate_median_in_cell(self, cell_data):
        cell_data.calculated["median"] = _get_median_from_list(cell_data.list_v, self.N_median)
        log.debug("_recalculate_median_in_cell: rect = {} median = {}".format(
            cell_data.rect, cell_data.calculated["median"]))

    def _recalculate_medians(self):
        for i, j in sorted(self.cells_data.keys()):
            log.debug("_recalculate_medians: (i,j)={}".format((i, j)))
            cell_data = self.cells_data[(i, j)]
            self._recalculate_median_in_cell(cell_data)

    def _drop_last_motion_in_cells(self):
        for i, j in sorted(self.cells_data.keys()):
            log.debug("_drop_last_motion_in_cells: (i,j)={}".format((i, j)))
            cell_data = self.cells_data[(i, j)]
            del cell_data.list_v[-1]
            log.debug("now len(list_v) = {}".format(len(cell_data.list_v)))

    def handle_image(self, frame, prev_frame):
        """
        The method receives the current frame and the previous frame
        and updates the field total_v that stores the estimates regular motion
        on the last frames of the video sequence.
        """
        begin_work_time = datetime.datetime.now()
        assert frame.shape == prev_frame.shape

        img_to_show = frame.copy()
        prev_img_to_show = prev_frame.copy()

        num_cells_with_motions = 0
        for i, j in sorted(self.cells_data.keys()):
            log.debug("handle_image: (i,j)={}".format((i, j)))
            cell_data = self.cells_data[(i, j)]
            self._handle_cell(cell_data, frame, prev_frame)

            rect = cell_data.rect
            v = cell_data.list_v[-1]
            if np.linalg.norm(v) >= self.min_motion:
                num_cells_with_motions += 1

            _draw_arrow(prev_img_to_show, _get_center(rect), v, (0, 255, 0))
            _draw_rect(prev_img_to_show, rect, color=(255, 0, 0), thickness=1)

        log.debug("num_cells_with_motions = {}".format(num_cells_with_motions))
        if num_cells_with_motions < len(self.cells_data) // 2:
            self._drop_last_motion_in_cells()
            self.total_v = None
            log.debug("total_v = {}".format(self.total_v))
            return img_to_show, prev_img_to_show
        self._recalculate_medians()

        list_medians = [np.array(cell_data.calculated["median"])
                        for cell_data in self.cells_data.values()
                        if cell_data.calculated.get("median")]
        log.debug("len(list_medians) = {}".format(len(list_medians)))
        list_medians = [v for v in list_medians if np.linalg.norm(v) >= self.min_motion]

        #
        # Idea for future development:
        # add check that most of the motions are directed like the median of them
        #

        log.debug("after filtering len(list_medians) = {}".format(len(list_medians)))
        if list_medians:
            list_medians = np.array(list_medians)
            log.debug("list_medians =\n%s", str(list_medians))
            total_v = np.median(list_medians, axis=0)
            total_v = Point(*total_v.tolist())
        else:
            total_v = Point(0, 0)

        self.total_v = total_v
        log.debug("total_v = {}".format(self.total_v))

        work_time = datetime.datetime.now() - begin_work_time
        work_time_ms = int(1000*work_time.total_seconds())
        log.debug("RoiMotionDetector.handle_image: work_time = {} ms".format(work_time_ms))

        return img_to_show, prev_img_to_show


def _get_subframe_for_motion(frame, vx, vy):
    def _get_subframe_from_tl(frame, vx, vy):
        assert vx >= 0 and vy >= 0
        h, w = frame.shape[:2]
        assert vx < w and vy < h
        return frame[: h - vy, : w - vx]
    def _get_subframe_from_br(frame, vx, vy):
        assert vx <= 0 and vy <= 0
        return frame[-vy:, -vx:]

    vx_p = int((vx + abs(vx)) / 2)
    vx_n = int((vx - abs(vx)) / 2)
    vy_p = int((vy + abs(vy)) / 2)
    vy_n = int((vy - abs(vy)) / 2)

    assert vx_p >= 0 and vy_p >= 0
    assert vx_n <= 0 and vy_n <= 0
    assert vx == vx_p + vx_n
    assert vy == vy_p + vy_n
    assert abs(vx_p * vx_n) < 1e-8 and abs(vy_p * vy_n) < 1e-8

    subframe = _get_subframe_from_tl(frame, vx_p, vy_p)
    subframe = _get_subframe_from_br(subframe, vx_n, vy_n)
    return subframe

def _move_mask_back_to_frame_size(mask, vx, vy, frame_shape):
    assert vx == int(vx)
    assert vy == int(vy)
    assert len(mask.shape) == 2

    mask_h, mask_w = mask.shape[:2]
    frame_h, frame_w = frame_shape[:2]

    assert frame_h == mask_h + abs(vy)
    assert frame_w == mask_w + abs(vx)

    if vx != 0:
        mask_horiz_shift = np.zeros((mask_h, abs(vx)), mask.dtype)
        if vx > 0:
            mask = np.hstack((mask_horiz_shift, mask))
        else:
            mask = np.hstack((mask, mask_horiz_shift))

    assert mask.shape[0] == mask_h
    assert mask.shape[1] == frame_w

    if vy != 0:
        mask_vert_shift = np.zeros((abs(vy), frame_w), mask.dtype)
        if vy > 0:
            mask = np.vstack((mask_vert_shift, mask))
        else:
            mask = np.vstack((mask, mask_vert_shift))

    assert mask.shape[:2] == frame_shape[:2]

    return mask

#pylint: disable=R0915
def _get_diff_as_mask(frame, prev_frame, total_v, blur_kernel_size=5, max_diff_to_be_same=10):
    begin_work_time = datetime.datetime.now()
    assert total_v is not None
    vx, vy = total_v
    vx = int(vx)
    vy = int(vy)

    prev_subframe = _get_subframe_for_motion(prev_frame, vx, vy)
    subframe = _get_subframe_for_motion(frame, -vx, -vy)

    prev_subframe_nomotion = _get_subframe_for_motion(prev_frame, -vx, -vy)

    assert subframe.shape == prev_subframe.shape
    subframe = subframe.astype(np.float32)
    prev_subframe = prev_subframe.astype(np.float32)
    prev_subframe_nomotion = prev_subframe_nomotion.astype(np.float32)
    _log_work_time("_get_diff_as_mask", "dt after subframes", begin_work_time)

    subframe = cv2.blur(subframe, (blur_kernel_size, blur_kernel_size))
    prev_subframe = cv2.blur(prev_subframe, (blur_kernel_size, blur_kernel_size))
    prev_subframe_nomotion = cv2.blur(prev_subframe_nomotion, (blur_kernel_size, blur_kernel_size))
    _log_work_time("_get_diff_as_mask", "dt after blur", begin_work_time)

    diff = (subframe - prev_subframe)
    diff_nomotion = (subframe - prev_subframe_nomotion)

    diff_min = np.amin(diff)
    diff_max = np.amax(diff)
    diff_to_show = (diff - diff_min) / (diff_max - diff_min)
    _log_work_time("_get_diff_as_mask", "dt after diff", begin_work_time)

    _dbg_imshow("prev_subframe", prev_subframe/255.)
    _dbg_imshow("subframe", subframe/255.)
    _dbg_imshow("diff", diff_to_show)
    min_diff_nomotion = np.amin(diff_nomotion)
    max_diff_nomotion = np.amax(diff_nomotion)
    _dbg_imshow("diff_nomotion",
                (diff_nomotion-min_diff_nomotion)/(max_diff_nomotion-min_diff_nomotion))

    absdiff = np.abs(diff)
    absdiff_nomotion = np.abs(diff_nomotion)
    assert absdiff.shape[2] == 3
    absdiff1 = absdiff[:, :, 0] + absdiff[:, :, 1] + absdiff[:, :, 2]
    absdiff_nomotion1 = (absdiff_nomotion[:, :, 0] + absdiff_nomotion[:, :, 1]
                         + absdiff_nomotion[:, :, 2])
    _log_work_time("_get_diff_as_mask", "dt after absdiff", begin_work_time)
    _dbg_imshow("absdiff1 from max", absdiff1 / np.amax(absdiff1))
    _dbg_imshow("absdiff_nomotion1 from max", absdiff_nomotion1 / np.amax(absdiff_nomotion1))

    assert absdiff_nomotion1.shape == absdiff1.shape

    mask1 = np.zeros(absdiff1.shape, np.float32)
    mask1[absdiff1 <= max_diff_to_be_same] = 1
    mask2 = np.zeros(absdiff1.shape, np.float32)
    mask2[absdiff_nomotion1 <= max_diff_to_be_same] = 1
    _dbg_imshow("mask1", mask1)
    _dbg_imshow("mask2", mask1)

    mask1[mask2 == 1] = 0
    _dbg_imshow("mask1-mask2", mask1)
    _log_work_time("_get_diff_as_mask", "dt after masking", begin_work_time)

    mask1 = _move_mask_back_to_frame_size(mask1, vx, vy, frame.shape)
    work_time = datetime.datetime.now() - begin_work_time
    work_time_ms = int(1000*work_time.total_seconds())
    log.debug("_get_diff_as_mask: work_time = {} ms".format(work_time_ms))
    return mask1

def _convert_connection_components(retval, labels, stats, centroids, original_mask):
    assert np.amax(labels) == retval - 1
    connected_components = [None] * retval
    for i in range(retval):
        mask = np.array(labels == i, dtype=np.uint8)
        stat_for_label = stats[i]
        stat_left = stat_for_label[cv2.CC_STAT_LEFT]
        stat_top = stat_for_label[cv2.CC_STAT_TOP]
        stat_width = stat_for_label[cv2.CC_STAT_WIDTH]
        stat_height = stat_for_label[cv2.CC_STAT_HEIGHT]
        rect = Rect(stat_left, stat_top, stat_width, stat_height)
        centr = centroids[i]
        area = _get_area_rect(rect)
        num = int(np.sum(original_mask[mask == 1]))
        if area > labels.shape[0] * labels.shape[1] / 16:
            log.debug("_convert_connection_components: i = {}".format(i))
            log.debug("_convert_connection_components: rect = {}".format(rect))
            log.debug("_convert_connection_components: centr = {}".format(centr))
            log.debug("_convert_connection_components: area = {}".format(area))
            log.debug("_convert_connection_components: num = {}".format(num))

        component = ConnectedComponent(label_id=i, mask=mask, centroid=centr, rect=rect,
                                       area=area, num=num)
        connected_components[i] = component

    return connected_components


def _find_threshold(average_mask, min_quantile=0.6):
    assert 0 < min_quantile < 0.99

    cur_min = np.amin(average_mask)
    cur_max = np.amax(average_mask)
    if cur_min == cur_max:
        return cur_min

    hist, bin_edges = np.histogram(average_mask, 20)

    assert bin_edges[0] >= 0, "bin_edges={}, min={}, max={}".format(
        bin_edges, np.amin(average_mask), np.amax(average_mask))

    total_num_el = average_mask.shape[0] * average_mask.shape[1]
    num_vals_lt_cur_bin_edge = 0
    for i in range(1, len(bin_edges)):
        cur_bin_edge = bin_edges[i]
        assert cur_bin_edge > 0
        num_vals_lt_cur_bin_edge += hist[i-1]
        if num_vals_lt_cur_bin_edge >= min_quantile * total_num_el:
            return cur_bin_edge

    raise RuntimeError("Error of the algorithm -- wrong work with histogram")


def _find_best_bbox_from_motion_mask(average_motion_mask,
                                     quantile=0.6,
                                     max_num_of_best_masks_to_unite=10,
                                     desired_rel_num_pixels_in_united_mask=0.3):
    begin_work_time = datetime.datetime.now()

    quantile_edge = _find_threshold(average_motion_mask, quantile)
    thresholded_mask = np.array(average_motion_mask >= quantile_edge).astype(np.uint8)
    thresholded_mask_to_show = cv2.cvtColor(thresholded_mask*255, cv2.COLOR_GRAY2BGR)
    _dbg_imshow("thresholded mask", thresholded_mask_to_show)
    log.debug("total_el_in mask = {}".format(
        average_motion_mask.shape[0] * average_motion_mask.shape[1]))
    log.debug("num_el gt quantile_edge in mask = {}".format(
        np.transpose(np.nonzero(average_motion_mask >= quantile_edge)).shape))
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded_mask)

    connected_components = _convert_connection_components(retval, labels, stats, centroids,
                                                          thresholded_mask)

    connected_components_sorted_by_num = sorted(connected_components, key=lambda c: -int(c.num))

    for ii in range(min(max_num_of_best_masks_to_unite+2, len(connected_components_sorted_by_num))):
        # this cycle is for debugging only
        cur_component = connected_components_sorted_by_num[ii]
        log.debug("connected_components_sorted_by_num[{}] = {}".format(ii, cur_component))
        _dbg_imshow("conn component ii="+str(ii), cur_component.mask * 255)

    desired_num = int(average_motion_mask.shape[0]
                      * average_motion_mask.shape[1]
                      * desired_rel_num_pixels_in_united_mask)
    best_components = []
    sum_best_components_num = 0
    log.debug("scanning connected components: desired_num = {}".format(desired_num))
    for ii in range(min(max_num_of_best_masks_to_unite, len(connected_components_sorted_by_num))):
        log.debug("scanning connected components: ii = {}".format(ii))
        cur_component = connected_components_sorted_by_num[ii]
        best_components.append(cur_component)
        log.debug("scanning connected components: cur_component.num = {}".format(cur_component.num))
        sum_best_components_num += cur_component.num
        log.debug("scanning connected components: sum_best_components_num = {}".format(
            sum_best_components_num))
        if sum_best_components_num >= desired_num:
            break


    if not best_components:
        return None

    res_bbox = best_components[0].rect
    for c in best_components[1:]:
        res_bbox = _get_union_rects(res_bbox, c.rect)

    best_component_to_show = cv2.cvtColor(best_components[0].mask*255, cv2.COLOR_GRAY2BGR)
    for c in best_components[1:]:
        best_component_to_show += cv2.cvtColor(c.mask*255, cv2.COLOR_GRAY2BGR)

    for c in best_components:
        _draw_rect(best_component_to_show, c.rect, (255, 0, 0), 3)

    _my_imshow("best_component", best_component_to_show)
    _log_work_time("_find_best_bbox_from_motion_mask", "work_time", begin_work_time)

    return res_bbox

#pylint: disable=R0902,R0913,C0111
class RoiDetectorImpl:
    """
    Implementation class -- allows to detect ROI on frames received from a video sequence
    using motion detection + background subtraction approaches.
    """
    def __init__(self, cell_params, increase_rect_params, N_median, min_motion_to_work,
                 max_num_of_best_masks_to_unite=5,
                 desired_rel_num_pixels_in_united_mask=0.3,
                 required_num_motions_in_last_frames=5,
                 num_last_frames_to_count_motions=1000):

        self.frame_size = None # (w, h)
        self.motion_detector = None
        self.cell_params = cell_params
        self.increase_rect_params = increase_rect_params
        self.N_median = N_median
        self.min_motion_to_work = min_motion_to_work

        self.sum_motion_masks = None
        self.num_summed_masks = 0
        self.res_bbox = None
        self.res_bbox_confidence = 0

        self.work_times_in_sec = []
        self.max_num_work_times = 10000

        self.rel_threshold_for_center = 0.9
        self.quantile_for_best_bbox = 0.5
        self.result_img_to_show = None

        self.max_num_of_best_masks_to_unite = max_num_of_best_masks_to_unite
        self.desired_rel_num_pixels_in_united_mask = desired_rel_num_pixels_in_united_mask

        self.required_num_motions_in_last_frames = required_num_motions_in_last_frames
        self.num_last_frames_to_count_motions = num_last_frames_to_count_motions

        self.last_frame_ids_with_motions = []

    #pylint: disable=R0914
    def handle_frame(self, frame, prev_frame, frame_id):
        """
        The method receives frame, the previous frame, and the frame number, and returns
        roi where there is some regular motion on several last frames of the video sequence.
        """
        if self.frame_size is None:
            h, w = frame.shape[:2]
            self.frame_size = (w, h)
            self.motion_detector = RoiMotionDetector(self.frame_size[::-1], self.cell_params,
                                                     self.increase_rect_params, self.N_median,
                                                     self.min_motion_to_work)

        assert self.frame_size is not None
        assert self.motion_detector is not None

        begin_work_time = datetime.datetime.now()

        assert frame.shape == prev_frame.shape
        assert tuple(frame.shape[:2]) == tuple(self.frame_size[::-1]), (
            "frame.shape[:2]={}, self.frame_size[::-1]={}".format(frame.shape[:2],
                                                                  self.frame_size[::-1]))
        assert isinstance(frame_id, int)
        assert (not self.last_frame_ids_with_motions
                or (self.last_frame_ids_with_motions[-1] < frame_id))

        self.res_bbox = None

        img_to_show, prev_img_to_show = self.motion_detector.handle_image(frame, prev_frame)
        self.result_img_to_show = prev_img_to_show

        total_v = self.motion_detector.total_v
        log.debug("main: total_v = {}".format(total_v))

        if total_v is not None:
            motion_mask = _get_diff_as_mask(frame, prev_frame, total_v)
            self.last_frame_ids_with_motions.append(frame_id)

            if self.num_summed_masks > 0:
                self.sum_motion_masks = self.sum_motion_masks + motion_mask
                self.num_summed_masks += 1
            else:
                self.sum_motion_masks = motion_mask.copy().astype(np.float32)
                self.num_summed_masks = 1

        # required to calculate res_bbox_confidence
        while (self.last_frame_ids_with_motions and
               (frame_id - self.last_frame_ids_with_motions[0]
                >
                self.num_last_frames_to_count_motions)):
            del self.last_frame_ids_with_motions[0]

        # simple approach to calculate res_bbox_confidence
        if len(self.last_frame_ids_with_motions) >= self.required_num_motions_in_last_frames:
            self.res_bbox_confidence = 1
        else:
            self.res_bbox_confidence = 0

        _dbg_imshow("frame", img_to_show)
        _dbg_imshow("prev frame", self.result_img_to_show)

        if self.num_summed_masks > 0:
            average_motion_mask = self.sum_motion_masks / self.num_summed_masks
            _dbg_imshow("average motion mask", average_motion_mask)

            self.res_bbox = _find_best_bbox_from_motion_mask(
                average_motion_mask=average_motion_mask,
                quantile=self.quantile_for_best_bbox,
                max_num_of_best_masks_to_unite=self.max_num_of_best_masks_to_unite,
                desired_rel_num_pixels_in_united_mask=self.desired_rel_num_pixels_in_united_mask)

        if self.res_bbox:
            conf_color = (200, 155, 0) if self.res_bbox_confidence == 1 else (255, 255, 55)
            _draw_rect(self.result_img_to_show, self.res_bbox, conf_color, 2)

        _my_imshow("result", self.result_img_to_show)

        work_time = datetime.datetime.now() - begin_work_time
        self.work_times_in_sec.append(work_time.total_seconds())
        while len(self.work_times_in_sec) > self.max_num_work_times:
            del self.work_times_in_sec[0]
        work_time_ms = int(1000*work_time.total_seconds())
        log.debug("work_time = {} ms".format(work_time_ms))
        avg_work_time_ms = int(1000*np.average(self.work_times_in_sec))
        log.debug("avg work_time = {} ms".format(avg_work_time_ms))
        return self.res_bbox

    def get_res_bbox(self):
        return self.res_bbox
    def get_res_bbox_confidence(self):
        """ At the moment it either 0 or 1 """
        return self.res_bbox_confidence
    def get_num_summed_masks(self):
        return self.num_summed_masks
    def get_result_img_to_show(self):
        return self.result_img_to_show


#pylint: disable=R0903
class RoiDetector:
    """
    The class allows to detect ROI on frames received from a video sequence
    using motion detection + background subtraction approaches.
    """
    @staticmethod
    def _create_default_roi_detector_impl(desired_min_side):
        increase_cell_coeff = 1.4
        shift_x = 1
        shift_y = 1
        increase_rect_params = IncreaseRectParams(increase_cell_coeff, shift_x, shift_y)

        grid_cell_size = int(25.0 / 160.0 * desired_min_side)
        cell_aspect_ratio = 1
        num_cells_x = 3
        num_cells_y = 3
        cell_params = CellParams(cell_height=grid_cell_size,
                                 cell_aspect_ratio=cell_aspect_ratio,
                                 cell_overlap=0, num_cells_x=num_cells_x,
                                 num_cells_y=num_cells_y, list_v_len=100)

        N_median = 20
        min_motion_to_work = 1.5 / 160.0 * desired_min_side
        roi_detector_impl = RoiDetectorImpl(cell_params, increase_rect_params,
                                            N_median, min_motion_to_work)
        return roi_detector_impl

    def __init__(self, frame_step):
        """
        Constructor.
        The only parameter of the constructor is frame step that should be used during detection.
        The value depends on the frome rate of the input video.
        The recommended value for video stream with frame rate 30 frames per second is frame_step=5.
        """
        self.frame_step = frame_step

        # this is the most important metric parameter
        # it depends on the quality of the video
        self.desired_min_side = 160

        self.max_frames_keep_bbox = 50
        self.max_len_bboxes_list = 100

        self.impl = self._create_default_roi_detector_impl(self.desired_min_side)

        self.frame_idx = -1
        self.prev_frame = None
        self.last_frame_detected_bbox = None
        self.detected_bboxes_for_avg = []
        self.scale = None

    @staticmethod
    def _prepare_frame_for_default_roi_detector(frame, desired_min_side):
        h, w = frame.shape[:2]
        min_sz = min(h, w)
        scale = float(desired_min_side) / min_sz
        target_size = (int(w * scale), int(h * scale))
        scaled_frame = cv2.resize(frame, target_size)
        return scaled_frame, scale

    def handle_frame(self, frame):
        """
        The main method of the class.
        The frames should be passed to the method with a constant frame rate
        (~30 frames per second).

        The method returns
        * either bounding box of detected ROI,
          (in this case it returns bounding box as namedtuple Rect),
        * or None if it cannot make detection with sufficient confidence.
        """

        self.frame_idx += 1
        log.debug("frame_idx = {}".format(self.frame_idx))

        cur_frame, scale = self._prepare_frame_for_default_roi_detector(
            frame, self.desired_min_side)
        assert cur_frame is not None
        assert scale > 0

        if self.scale is None:
            self.scale = scale
        assert self.scale == scale

        if self.prev_frame is None:
            assert self.frame_idx == 0
            self.prev_frame = cur_frame.copy()
            log.debug("return None as self.prev_frame is None")
            return None


        if self.frame_idx % self.frame_step == 0:
            detected_bbox = self.impl.handle_frame(cur_frame, self.prev_frame, self.frame_idx)
            detected_bbox_confidence = self.impl.get_res_bbox_confidence()
            if detected_bbox_confidence > 0:
                self.last_frame_detected_bbox = self.frame_idx

                self.detected_bboxes_for_avg.append(detected_bbox)
                while len(self.detected_bboxes_for_avg) > self.max_len_bboxes_list:
                    del self.detected_bboxes_for_avg[0]

            self.prev_frame = cur_frame.copy()
        else:
            log.debug("skipping frame")


        should_return_bbox = (
            (self.last_frame_detected_bbox is not None) and
            (self.frame_idx - self.last_frame_detected_bbox) < self.max_frames_keep_bbox)

        if not should_return_bbox:
            return None

        avg_bbox = _get_median_of_rects(self.detected_bboxes_for_avg)
        rescaled_bbox = _scale_rect(avg_bbox, 1.0 / self.scale)
        return rescaled_bbox
