"""
Copyright (c) 2018-2024 Intel Corporation

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

from ..adapters import Adapter
from ..config import ConfigValidator, StringField, NumberField
from ..representation import TextDetectionPrediction
from ..utils import UnsupportedPackage

try:
    import pyclipper
except ImportError as import_error:
    pyclipper = UnsupportedPackage('pyclipper', import_error.msg)

try:
    from shapely.geometry import Polygon
except ImportError as import_error:
    Polygon = UnsupportedPackage("shapely", import_error.msg)


class TextDetectionAdapter(Adapter):
    __provider__ = 'pixel_link_text_detection'
    prediction_types = (TextDetectionPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'pixel_link_out': StringField(
                description="Name of layer containing information related to linkage "
                            "between pixels and their neighbors."
            ),
            'pixel_class_out': StringField(
                description="Name of layer containing information related to "
                            "text/no-text classification for each pixel."
            ),
            'pixel_class_confidence_threshold': NumberField(
                description='confidence threshold for valid segmentation mask',
                optional=True, default=0.8, value_type=float, min_value=0, max_value=1
            ),
            'pixel_link_confidence_threshold': NumberField(
                description='confidence threshold for valid pixel links',
                optional=True, default=0.8, value_type=float, min_value=0, max_value=1
            ),
            'min_area': NumberField(
                value_type=int, min_value=0, default=0, optional=True,
                description='minimal area for valid text prediction'
            ),
            'min_height': NumberField(
                value_type=int, min_value=0, default=0, optional=True,
                description='minimal height for valid text prediction'
            )
        })

        return parameters

    @classmethod
    def validate_config(cls, config, fetch_only=False, **kwargs):
        return super().validate_config(
            config, fetch_only=fetch_only, on_extra_argument=ConfigValidator.WARN_ON_EXTRA_ARGUMENT
        )

    def configure(self):
        self.pixel_link_out = self.get_value_from_config('pixel_link_out')
        self.pixel_class_out = self.get_value_from_config('pixel_class_out')
        self.pixel_link_confidence_threshold = self.get_value_from_config('pixel_link_confidence_threshold')
        self.pixel_class_confidence_threshold = self.get_value_from_config('pixel_class_confidence_threshold')
        self.min_area = self.get_value_from_config('min_area')
        self.min_height = self.get_value_from_config('min_height')
        self.outputs_verified = False

    def select_output_blob(self, outputs):
        self.pixel_link_out = self.check_output_name(self.pixel_link_out, outputs)
        self.pixel_class_out = self.check_output_name(self.pixel_class_out, outputs)
        self.outputs_verified = True

    def process(self, raw, identifiers, frame_meta):
        results = []
        predictions = self._extract_predictions(raw, frame_meta)
        if not self.outputs_verified:
            self.select_output_blob(predictions)

        def _input_parameters(input_meta):
            input_shape = next(iter(input_meta.get('input_shape').values()))
            original_image_size = input_meta.get('image_size')
            layout = 'NCHW' if input_shape[1] == original_image_size[2] else 'NHWC'

            return original_image_size, layout
        raw_output = zip(identifiers, frame_meta, predictions[self.pixel_link_out], predictions[self.pixel_class_out])
        for identifier, current_frame_meta, link_data, cls_data in raw_output:
            image_size, layout = _input_parameters(current_frame_meta)
            if layout == 'NCHW':
                link_data = link_data.transpose((1, 2, 0))
                cls_data = cls_data.transpose((1, 2, 0))
            new_link_data = link_data.reshape([*link_data.shape[:2], 8, 2])
            new_link_data = self.softmax(new_link_data)
            cls_data = self.softmax(cls_data)
            decoded_rects = self.to_boxes(image_size, cls_data[:, :, 1], new_link_data[:, :, :, 1])
            results.append(TextDetectionPrediction(identifier, decoded_rects))

        return results

    def mask_to_bboxes(self, mask, image_shape):
        """ Converts mask to bounding boxes. """

        def rect_to_xys(rect, image_shape):
            """ Converts rotated rectangle to points. """

            height, width = image_shape[0:2]

            def get_valid_x(x_coord):
                return np.clip(x_coord, 0, width - 1)

            def get_valid_y(y_coord):
                return np.clip(y_coord, 0, height - 1)

            rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
            points = cv2.boxPoints(rect)
            points = points.astype(np.intp)
            for i_xy, (x_coord, y_coord) in enumerate(points):
                x_coord = get_valid_x(x_coord)
                y_coord = get_valid_y(y_coord)
                points[i_xy, :] = [x_coord, y_coord]

            return points

        def min_area_rect(contour):
            """ Returns minimum area rectangle. """

            (center_x, center_y), (width, height), theta = cv2.minAreaRect(contour)
            return [center_x, center_y, width, height, theta], width * height

        image_h, image_w = image_shape[0:2]

        bboxes = []
        max_bbox_idx = mask.max()
        mask = cv2.resize(mask, (image_w, image_h), interpolation=cv2.INTER_NEAREST)

        for bbox_idx in range(1, max_bbox_idx + 1):
            bbox_mask = (mask == bbox_idx).astype(np.uint8)
            cnts = cv2.findContours(bbox_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2]

            if not cnts:
                continue

            cnt = cnts[0]
            rect, rect_area = min_area_rect(cnt)

            box_width, box_height = rect[2:-1]
            if min(box_width, box_height) < self.min_height:
                continue

            if rect_area < self.min_area:
                continue

            xys = rect_to_xys(rect, image_shape)
            bboxes.append(xys)

        return bboxes

    @staticmethod
    def softmax(logits):
        """ Returns softmax given logits. """

        max_logits = np.max(logits, axis=-1, keepdims=True)
        numerator = np.exp(logits - max_logits)
        denominator = np.sum(numerator, axis=-1, keepdims=True)

        return numerator / denominator

    def to_boxes(self, image_shape, segm_pos_scores, link_pos_scores):
        """ Returns boxes for each image in batch. """

        mask = self.decode_image(segm_pos_scores, link_pos_scores)
        mask = np.asarray(mask, np.int32)[...]
        bboxes = self.mask_to_bboxes(mask, image_shape)

        return bboxes

    def decode_image(self, segm_scores, link_scores):
        """ Convert softmax scores to mask. """

        segm_mask = segm_scores >= self.pixel_class_confidence_threshold
        link_mask = link_scores >= self.pixel_link_confidence_threshold
        points = list(zip(*np.where(segm_mask)))
        height, width = np.shape(segm_mask)
        group_mask = dict.fromkeys(points, -1)

        def find_parent(point):
            return group_mask[point]

        def set_parent(point, parent):
            group_mask[point] = parent

        def is_root(point):
            return find_parent(point) == -1

        def find_root(point):
            root = point
            update_parent = False
            while not is_root(root):
                root = find_parent(root)
                update_parent = True

            if update_parent:
                set_parent(point, root)

            return root

        def join(point1, point2):
            root1 = find_root(point1)
            root2 = find_root(point2)

            if root1 != root2:
                set_parent(root1, root2)

        def get_neighbours(x_coord, y_coord):
            """ Returns 8-point neighbourhood of given point. """

            return [
                (x_coord - 1, y_coord - 1), (x_coord, y_coord - 1), (x_coord + 1, y_coord - 1),
                (x_coord - 1, y_coord), (x_coord + 1, y_coord),
                (x_coord - 1, y_coord + 1), (x_coord, y_coord + 1), (x_coord + 1, y_coord + 1)
            ]

        def is_valid_coord(x_coord, y_coord, width, height):
            """ Returns true if given point inside image frame. """
            return 0 <= x_coord < width and 0 <= y_coord < height

        def get_all():
            root_map = {}

            def get_index(root):
                if root not in root_map:
                    root_map[root] = len(root_map) + 1
                return root_map[root]

            mask = np.zeros_like(segm_mask, dtype=np.int32)
            for point in points:
                point_root = find_root(point)
                bbox_idx = get_index(point_root)
                mask[point] = bbox_idx
            return mask

        for point in points:
            y_coord, x_coord = point
            neighbours = get_neighbours(x_coord, y_coord)
            for n_idx, (neighbour_x, neighbour_y) in enumerate(neighbours):
                if is_valid_coord(neighbour_x, neighbour_y, width, height):
                    link_value = link_mask[y_coord, x_coord, n_idx]
                    segm_value = segm_mask[neighbour_y, neighbour_x]
                    if link_value and segm_value:
                        join(point, (neighbour_y, neighbour_x))

        mask = get_all()
        return mask


class EASTTextDetectionAdapter(Adapter):
    __provider__ = 'east_text_detection'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'score_map_out': StringField(description='name of layer with score map'),
            'geometry_map_out': StringField(description='name of layer with geometry map'),
            'score_map_threshold': NumberField(
                value_type=float, optional=True, default=0.8, min_value=0, description='threshold for scores map'
            ),
            'nms_threshold': NumberField(value_type=float, min_value=0, default=0.2, description='threshold for NMS'),
            'box_threshold': NumberField(value_type=float, min_value=0, default=0.1, description='threshold for boxes')
        })
        return parameters

    def configure(self):
        self.score_map_out = self.get_value_from_config('score_map_out')
        self.geometry_map_out = self.get_value_from_config('geometry_map_out')
        self.score_map_thresh = self.get_value_from_config('score_map_threshold')
        self.nms_thresh = self.get_value_from_config('nms_threshold')
        self.box_thresh = self.get_value_from_config('box_threshold')
        if isinstance(Polygon, UnsupportedPackage):
            Polygon.raise_error(self.__provider__)
        self.outputs_verified = False

    def select_output_blob(self, outputs):
        self.score_map_out = self.check_output_name(self.score_map_out, outputs)
        self.geometry_map_out = self.check_output_name(self.geometry_map_out, outputs)
        self.outputs_verified = True

    def process(self, raw, identifiers, frame_meta):
        raw_outputs = self._extract_predictions(raw, frame_meta)
        if not self.outputs_verified:
            self.select_output_blob(raw_outputs)
        score_maps = raw_outputs[self.score_map_out]
        geometry_maps = raw_outputs[self.geometry_map_out]
        is_nchw = score_maps.shape[1] == 1
        results = []
        if is_nchw:
            score_maps = np.transpose(score_maps, (0, 2, 3, 1))
            geometry_maps = np.transpose(geometry_maps, (0, 2, 3, 1))
        for identifier, score_map, geo_map, meta in zip(identifiers, score_maps, geometry_maps, frame_meta):
            if len(score_map.shape) == 3:
                score_map = score_map[:, :, 0]
                geo_map = geo_map[:, :, ]
            xy_text = np.argwhere(score_map > self.score_map_thresh)
            xy_text = xy_text[np.argsort(xy_text[:, 0])]
            text_box_restored = self.restore_rectangle(
                xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :]
            )  # N*4*2
            boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
            boxes[:, :8] = text_box_restored.reshape((-1, 8))
            boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]

            boxes = self.nms_locality(boxes.astype('float32'), self.nms_thresh)

            if boxes.shape[0] == 0:
                results.append(TextDetectionPrediction(identifier, boxes))
                continue
            for i, box in enumerate(boxes):
                mask = np.zeros_like(score_map, dtype=np.uint8)
                cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
                boxes[i, 8] = cv2.mean(score_map, mask)[0]
            boxes = boxes[boxes[:, 8] > self.box_thresh]

            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= meta.get('scale_x', 1)
            boxes[:, :, 1] /= meta.get('scale_y', 1)
            results.append(TextDetectionPrediction(identifier, boxes))

        return results

    @staticmethod
    def nms_locality(polys, thres=0.3):
        '''
        locality aware nms of EAST
        :param polys: a N*9 numpy array. first 8 coordinates, then prob
        :return: boxes after nms
        '''
        def intersection(g, p):
            g = Polygon(g[:8].reshape((4, 2)))
            p = Polygon(p[:8].reshape((4, 2)))
            if not g.is_valid or not p.is_valid:
                return 0
            inter = Polygon(g).intersection(Polygon(p)).area
            union = g.area + p.area - inter
            if union == 0:
                return 0
            return inter / union

        def weighted_merge(g, p):
            g[:8] = (g[8] * g[:8] + p[8] * p[:8]) / (g[8] + p[8])
            g[8] = (g[8] + p[8])
            return g

        def standard_nms(S, thres):
            order = np.argsort(S[:, 8])[::-1]
            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])

                inds = np.where(ovr <= thres)[0]
                order = order[inds + 1]
            return S[keep]
        S = []
        p = None
        for g in polys:
            if p is not None and intersection(g, p) > thres:
                p = weighted_merge(g, p)
            else:
                if p is not None:
                    S.append(p)
                p = g
        if p is not None:
            S.append(p)

        if not S:
            return np.array([])

        return standard_nms(np.array(S), thres)

    @staticmethod
    def restore_rectangle(origin, geometry):
        d = geometry[:, :4]
        angle = geometry[:, 4]
        # for angle > 0
        origin_0 = origin[angle >= 0]
        d_0 = d[angle >= 0]
        angle_0 = angle[angle >= 0]
        if origin_0.shape[0] > 0:
            p = np.array([np.zeros(d_0.shape[0]), -d_0[:, 0] - d_0[:, 2],
                          d_0[:, 1] + d_0[:, 3], -d_0[:, 0] - d_0[:, 2],
                          d_0[:, 1] + d_0[:, 3], np.zeros(d_0.shape[0]),
                          np.zeros(d_0.shape[0]), np.zeros(d_0.shape[0]),
                          d_0[:, 3], -d_0[:, 2]])
            p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

            rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
            rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

            rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
            rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

            p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
            p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

            p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

            p3_in_origin = origin_0 - p_rotate[:, 4, :]
            new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
            new_p1 = p_rotate[:, 1, :] + p3_in_origin
            new_p2 = p_rotate[:, 2, :] + p3_in_origin
            new_p3 = p_rotate[:, 3, :] + p3_in_origin

            new_p_0 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                      new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
        else:
            new_p_0 = np.zeros((0, 4, 2))
        # for angle < 0
        origin_1 = origin[angle < 0]
        d_1 = d[angle < 0]
        angle_1 = angle[angle < 0]
        if origin_1.shape[0] > 0:
            p = np.array([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2],
                          np.zeros(d_1.shape[0]), -d_1[:, 0] - d_1[:, 2],
                          np.zeros(d_1.shape[0]), np.zeros(d_1.shape[0]),
                          -d_1[:, 1] - d_1[:, 3], np.zeros(d_1.shape[0]),
                          -d_1[:, 1], -d_1[:, 2]])
            p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

            rotate_matrix_x = np.array([np.cos(-angle_1), -np.sin(-angle_1)]).transpose((1, 0))
            rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

            rotate_matrix_y = np.array([np.sin(-angle_1), np.cos(-angle_1)]).transpose((1, 0))
            rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

            p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
            p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

            p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

            p3_in_origin = origin_1 - p_rotate[:, 4, :]
            new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
            new_p1 = p_rotate[:, 1, :] + p3_in_origin
            new_p2 = p_rotate[:, 2, :] + p3_in_origin
            new_p3 = p_rotate[:, 3, :] + p3_in_origin

            new_p_1 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                      new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
        else:
            new_p_1 = np.zeros((0, 4, 2))

        return np.concatenate([new_p_0, new_p_1])


class CRAFTTextDetectionAdapter(Adapter):
    __provider__ = 'craft_text_detection'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'score_out': StringField(description='name of layer with score map', optional=True),
            'text_threshold': NumberField(
                value_type=float, optional=True, default=0.7, min_value=0, description='text confidence threshold'
            ),
            'link_threshold': NumberField(
                value_type=float, optional=True, default=0.4, min_value=0, description='link confidence threshold'
            ),
            'low_text': NumberField(
                value_type=float, optional=True, default=0.4, min_value=0, description='text low-bound score'
            )
        })
        return parameters

    def configure(self):
        self.score_out = self.get_value_from_config('score_out')
        self.text_threshold = self.get_value_from_config('text_threshold')
        self.link_threshold = self.get_value_from_config('link_threshold')
        self.low_text = self.get_value_from_config('low_text')

    def process(self, raw, identifiers, frame_meta):
        raw_outputs = self._extract_predictions(raw, frame_meta)
        self.select_output_blob(raw_outputs)
        score_out = raw_outputs[self.score_out] if self.score_out else raw_outputs[self.output_blob]
        results = []
        for identifier, score, meta in zip(identifiers, score_out, frame_meta):
            score_text = score[:, :, 0]
            score_link = score[:, :, 1]

            boxes = self.get_detection_boxes(score_text, score_link,
                                             self.text_threshold, self.link_threshold, self.low_text)
            boxes = self.adjust_result_coordinates(boxes, meta.get('scale', 1.0))
            results.append(TextDetectionPrediction(identifier, boxes))

        return results

    @staticmethod
    def get_detection_boxes(text, link, text_threshold, link_threshold, low_text):
        img_h, img_w = text.shape

        _, score_text = cv2.threshold(text.copy(), low_text, 1, 0)
        _, score_link = cv2.threshold(link.copy(), link_threshold, 1, 0)

        text_score_comb = np.clip(score_text + score_link, 0, 1)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)

        det = []
        for k in range(1, count):
            # size filtering
            size = stats[k, cv2.CC_STAT_AREA]
            if size < 10:
                continue

            # thresholding
            if np.max(text[labels == k]) < text_threshold:
                continue

            # make segmentation map
            segmap = np.zeros(text.shape, dtype=np.uint8)
            segmap[labels == k] = 255
            segmap[np.logical_and(score_link == 1, score_text == 0)] = 0  # remove link area
            x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
            w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
            niter = int(np.sqrt(size * min(w, h) / (w * h)) * 2)
            sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
            # boundary check
            sx = max(sx, 0)
            sy = max(sy, 0)
            ex = min(ex, img_w)
            ey = min(ey, img_h)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
            segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

            # make box
            np_contours = np.roll(np.array(np.where(segmap != 0)), 1, axis=0).transpose().reshape(-1, 2)
            rectangle = cv2.minAreaRect(np_contours)
            box = cv2.boxPoints(rectangle)

            # align diamond-shape
            w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
            box_ratio = max(w, h) / (min(w, h) + 1e-5)
            if abs(1 - box_ratio) <= 0.1:
                l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
                t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
                box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

            # make clock-wise order
            startidx = box.sum(axis=1).argmin()
            box = np.roll(box, 4 - startidx, 0)
            box = np.array(box)

            det.append(box)

        return det

    @staticmethod
    def adjust_result_coordinates(polys, scale, ratio_net=2):
        polys = np.array(polys)
        for k, _ in enumerate(polys):
            polys[k] *= (scale * ratio_net, scale * ratio_net)
        return polys


class PPOCRDetectionAdapter(Adapter):
    __provider__ = 'ppocr_det'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'threshold': NumberField(
                optional=True, value_type=float, min_value=0, max_value=1, default=0.3,
                description='threshold for segmentation mask'),
            'box_threshold': NumberField(optional=True, default=0.7, min_value=0, max_value=1, value_type=float,
                                         description='boxes score threshold'
                                         ),
            'max_candidates': NumberField(optional=True, default=1000, value_type=int,
                                          description='maximum candidates'),
            'unclip_ratio': NumberField(optional=True, default=2.0, value_type=float, description='unclip ratio'),
            'min_size': NumberField(optional=True, value_type=float, default=3, min_value=0, description='min box size')
        })

        return params

    def configure(self):
        self.threshold = self.get_value_from_config('threshold')
        self.box_threshold = self.get_value_from_config('box_threshold')
        self.unclip_ratio = self.get_value_from_config('unclip_ratio')
        self.min_size = self.get_value_from_config('min_size')
        self.max_candidates = self.get_value_from_config('max_candidates')
        if isinstance(Polygon, UnsupportedPackage):
            Polygon.raise_error(self.__provider__)
        if isinstance(pyclipper, UnsupportedPackage):
            pyclipper.raise_error(self.__provider__)

    def process(self, raw, identifiers, frame_meta):
        raw_outputs = self._extract_predictions(raw, frame_meta)
        results = []
        for identifier, out, meta in zip(identifiers, raw_outputs[self.output_blob], frame_meta):
            src_h, src_w = meta['image_size'][:2]
            mask = out[0, :, :] > self.threshold
            boxes = self.boxes_from_bitmap(out[0, :, :], mask, src_w, src_h)
            results.append(TextDetectionPrediction(identifier, boxes))

        return results

    def boxes_from_bitmap(self, pred, bitmap, dest_width, dest_height):
        height, width = bitmap.shape
        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            _, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]
        num_contours = min(len(contours), self.max_candidates)
        boxes = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_threshold > score:
                continue
            box = self.unclip(points, self.unclip_ratio).reshape((-1, 1, 2))
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int16))
        return np.array(boxes, dtype=np.int16)

    @staticmethod
    def unclip(box, unclip_ratio):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    @staticmethod
    def get_mini_boxes(contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(cv2.boxPoints(bounding_box), key=lambda x: x[0])
        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1, index_4 = 0, 1
        else:
            index_1, index_4 = 1, 0
        if points[3][1] > points[2][1]:
            index_2, index_3 = 2, 3
        else:
            index_2, index_3 = 3, 2
        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    @staticmethod
    def box_score_fast(bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
