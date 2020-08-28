"""
Copyright (c) 2018-2020 Intel Corporation

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
from ..postprocessor import NMS
try:
    from shapely.geometry import Polygon
except ImportError:
    Polygon = None


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

    def validate_config(self):
        super().validate_config(on_extra_argument=ConfigValidator.WARN_ON_EXTRA_ARGUMENT)

    def configure(self):
        self.pixel_link_out = self.get_value_from_config('pixel_link_out')
        self.pixel_class_out = self.get_value_from_config('pixel_class_out')
        self.pixel_link_confidence_threshold = self.get_value_from_config('pixel_link_confidence_threshold')
        self.pixel_class_confidence_threshold = self.get_value_from_config('pixel_class_confidence_threshold')
        self.min_area = self.get_value_from_config('min_area')
        self.min_height = self.get_value_from_config('min_height')

    def process(self, raw, identifiers, frame_meta):
        results = []
        predictions = self._extract_predictions(raw, frame_meta)

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
            points = points.astype(np.int0)
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
            if np.size(cnts) == 0:
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


class TextProposalsDetectionAdapter(Adapter):
    __provider__ = 'ctpn_text_detection'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update(
            {
                'cls_prob_out': StringField(description='output layer name with class probabilities'),
                'bbox_pred_out': StringField(description='output layer name with boxes'),
                'min_size': NumberField(
                    min_value=0.0, default=8, optional=True, description='min detected text proposal size'
                ),
                'min_ratio': NumberField(
                    min_value=0.000001, default=0.5, optional=True, description='min ratio for text line'
                ),
                'line_min_score': NumberField(
                    min_value=0, max_value=1, default=0.9, optional=True, description='min confidence for text line'
                ),
                'text_proposals_width': NumberField(
                    value_type=int, min_value=1, description='min width for text proposals', default=16,
                    optional=True
                ),
                'min_num_proposals': NumberField(
                    value_type=int, min_value=1, description='min number for text proposals', default=2,
                    optional=True
                ),
                'pre_nms_top_n': NumberField(
                    value_type=int, min_value=1, description='save top n proposals before nms applying', default=12000,
                    optional=True
                ),
                'post_nms_top_n': NumberField(
                    value_type=int, min_value=1, description='save top n proposals after nms applying', default=1000,
                    optional=True
                ),
                'nms_threshold': NumberField(
                    value_type=float, min_value=0, description='overlap threshold for NMS', default=0.7, optional=True
                )
            }
        )
        return parameters

    def configure(self):
        self.cls_prob_out = self.get_value_from_config('cls_prob_out')
        self.bbox_pred_out = self.get_value_from_config('bbox_pred_out')
        self.min_size = self.get_value_from_config('min_size')
        self.pre_nms_top_n = self.get_value_from_config('pre_nms_top_n')
        self.post_nms_top_n = self.get_value_from_config('post_nms_top_n')
        self.nms_threshold = self.get_value_from_config('nms_threshold')
        self.min_ratio = self.get_value_from_config('min_ratio')
        self.line_min_score = self.get_value_from_config('line_min_score')
        self.text_proposals_width = self.get_value_from_config('text_proposals_width')
        self.min_num_proposals = self.get_value_from_config('min_num_proposals')
        if Polygon is None:
            raise ValueError("east_text_detection adapter requires shapely, please install it")
        self.text_proposal_connector = TextProposalConnector()

    def process(self, raw, identifiers, frame_meta):
        raw_outputs = self._extract_predictions(raw, frame_meta)
        result = []
        data = zip(raw_outputs[self.bbox_pred_out], raw_outputs[self.cls_prob_out], frame_meta, identifiers)
        for bbox_pred, cls_prob, meta, identifier in data:
            input_shape = next(iter(meta['input_shape'].values()))
            if input_shape[1] == 3:
                cls_prob = np.transpose(cls_prob, (1, 2, 0))
                bbox_pred = np.transpose(bbox_pred, (1, 2, 0))
            scale_x, scale_y = meta['scale_x'], meta['scale_y']
            im_info = [meta['original_height'], meta['original_width'], min(scale_x, scale_y)]
            textsegs = self.proposal_layer(cls_prob, bbox_pred, im_info)
            scores = textsegs[:, 0]
            textsegs = textsegs[:, 1:5]
            textsegs[:, 0::2] /= scale_x
            textsegs[:, 1::2] /= scale_y
            boxes = self.detect(textsegs, scores[:, np.newaxis], [meta['original_height'], meta['original_width']])
            boxes = boxes[:, :8]
            geom_operations = meta['geometric_operations']
            resize_op = [geom_operation for geom_operation in geom_operations if geom_operation.type == 'resize']
            if len(resize_op) >= 2:
                scale_x, scale_y = resize_op[0].parameters['scale_y'], resize_op[0].parameters['scale_x']
                boxes[:, 0::2] /= scale_x
                boxes [:, 1::2] /= scale_y
            rects = [box.reshape(4, 2) for box in boxes]
            result.append(TextDetectionPrediction(identifier, np.array(rects)))

        return result

    def proposal_layer(self, rpn_cls_prob_reshape, rpn_bbox_pred, im_info, _feat_stride=(16, )):
        """
        Parameters
        rpn_cls_prob_reshape: (H , W , Ax2) outputs of RPN, prob of bg or fg
        rpn_bbox_pred: (H , W , Ax4), rgs boxes output of RPN
        im_info: a list of [image_height, image_width, scale_ratios]
        _feat_stride: the downsampling ratio of feature map to the original input image
        Algorithm:
        for each (H, W) location i
        generate A anchor boxes centered on cell i
        apply predicted bbox deltas at cell i to each of the A anchors
        clip predicted boxes to image
        remove predicted boxes with either height or width < threshold
        sort all (proposal, score) pairs by score from highest to lowest
        take top pre_nms_topN proposals before NMS
        apply NMS with threshold to remaining proposals
        take after_nms_top_n proposals after NMS
        return the top proposals (-> RoIs top, scores top)
        """

        def _filter_boxes(boxes, min_size):
            """Remove all boxes with any side smaller than min_size."""
            ws = boxes[:, 2] - boxes[:, 0] + 1
            hs = boxes[:, 3] - boxes[:, 1] + 1
            keep = np.where((ws >= min_size) & (hs >= min_size))[0]

            return keep

        _anchors = self.generate_anchors()
        _num_anchors = _anchors.shape[0]
        height, width = rpn_cls_prob_reshape.shape[:2]
        scores = np.reshape(
            np.reshape(rpn_cls_prob_reshape, [height, width, _num_anchors, 2])[:, :, :, 1],
            [height, width, _num_anchors]
        )

        bbox_deltas = rpn_bbox_pred
        shift_x = np.arange(0, width) * _feat_stride
        shift_y = np.arange(0, height) * _feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
        _num_shifts = shifts.shape[0]
        anchors = _anchors.reshape((1, _num_anchors, 4)) + shifts.reshape((1, _num_shifts, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((_num_shifts * _num_anchors, 4))
        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        # bbox deltas will be (4 * A, H, W) format
        # transpose to (H, W, 4 * A)
        # reshape to (H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = bbox_deltas.reshape((-1, 4))  # (HxWxA, 4)

        # Same story for the scores:
        scores = scores.reshape((-1, 1))

        # Convert anchors into proposals via bbox transformations
        proposals = self.bbox_transform_inv(anchors, bbox_deltas)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info[:2])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = _filter_boxes(proposals, self.min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep]
        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if self.pre_nms_top_n > 0:
            order = order[:self.pre_nms_top_n]
        proposals = proposals[order, :]
        scores = scores[order]
        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        keep = NMS.nms(
            proposals[:, 0], proposals[:, 1], proposals[:, 2], proposals[:, 3], scores.reshape(-1), self.nms_threshold
        )
        if self.post_nms_top_n > 0:
            keep = keep[:self.post_nms_top_n]
        proposals = proposals[keep, :]
        scores = scores[keep]
        blob = np.hstack((scores.astype(np.float32, copy=False), proposals.astype(np.float32, copy=False)))

        return blob

    @staticmethod
    def generate_anchors():
        def generate_basic_anchors(sizes, base_size=16):
            base_anchor = np.array([0, 0, base_size - 1, base_size - 1], np.int32)
            anchors = np.zeros((len(sizes), 4), np.int32)
            index = 0
            for h, w in sizes:
                anchors[index] = scale_anchor(base_anchor, h, w)
                index += 1
            return anchors

        def scale_anchor(anchor, h, w):
            x_ctr = (anchor[0] + anchor[2]) * 0.5
            y_ctr = (anchor[1] + anchor[3]) * 0.5
            scaled_anchor = anchor.copy()
            scaled_anchor[0] = x_ctr - w / 2  # xmin
            scaled_anchor[2] = x_ctr + w / 2  # xmax
            scaled_anchor[1] = y_ctr - h / 2  # ymin
            scaled_anchor[3] = y_ctr + h / 2  # ymax
            return scaled_anchor

        heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
        widths = [16]
        sizes = []
        for h in heights:
            for w in widths:
                sizes.append((h, w))
        return generate_basic_anchors(sizes)

    @staticmethod
    def bbox_transform_inv(boxes, deltas):

        boxes = boxes.astype(deltas.dtype, copy=False)

        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        dy = deltas[:, 1::4]
        dh = deltas[:, 3::4]

        pred_ctr_x = ctr_x[:, np.newaxis]
        pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
        pred_w = widths[:, np.newaxis]
        pred_h = np.exp(dh) * heights[:, np.newaxis]

        pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes

    def detect(self, text_proposals, scores, size):
        keep_inds = np.where(scores > 0.7)[0]
        text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]

        sorted_indices = np.argsort(scores.ravel())[::-1]
        text_proposals, scores = text_proposals[sorted_indices], scores[sorted_indices]
        x_mins, y_mins = text_proposals[:, 0], text_proposals[:, 1]
        x_maxs, y_maxs = text_proposals[:, 2], text_proposals[:, 3]

        keep_inds = NMS.nms(x_mins, y_mins, x_maxs, y_maxs, scores.reshape(-1), 0.2)
        text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]

        text_recs = self.text_proposal_connector.get_text_lines(text_proposals, scores, size)
        heights = np.zeros((len(text_recs), 1), np.float)
        widths = np.zeros((len(text_recs), 1), np.float)
        scores = np.zeros((len(text_recs), 1), np.float)
        index = 0
        for box in text_recs:
            heights[index] = (abs(box[5] - box[1]) + abs(box[7] - box[3])) / 2.0 + 1
            widths[index] = (abs(box[2] - box[0]) + abs(box[6] - box[4])) / 2.0 + 1
            scores[index] = box[8]
            index += 1
        keep_inds = np.where(
            (widths / heights > self.min_ratio) & (scores > self.line_min_score) &
            (widths > (self.min_num_proposals * self.text_proposals_width))
        )[0]

        return text_recs[keep_inds]


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)

    return boxes


class Graph:
    def __init__(self, graph):
        self.graph = graph

    def sub_graphs_connected(self):
        sub_graphs = []
        for index in range(self.graph.shape[0]):
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v = index
                sub_graphs.append([v])
                while self.graph[v, :].any():
                    v = np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)

        return sub_graphs


class TextProposalGraphBuilder:
    """
        Build Text proposals into a graph.
    """
    def get_successions(self, index):
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) + 1, min(int(box[0]) + 50 + 1, self.im_size[1])):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if results:
                return results
        return results

    def get_precursors(self, index):
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) - 1, max(int(box[0] - 50), 0) - 1, -1):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if results:
                return results
        return results

    def is_succession_node(self, index, succession_index):
        precursors = self.get_precursors(succession_index)
        if self.scores[index] >= np.max(self.scores[precursors]):
            return True
        return False

    def meet_v_iou(self, index1, index2):
        def overlaps_v(h1, h2, text_proposal1, text_proposal2):
            y0 = max(text_proposal2[1], text_proposal1[1])
            y1 = min(text_proposal2[3], text_proposal1[3])
            return max(0, y1 - y0 + 1) / min(h1, h2)

        def size_similarity(h1, h2):
            return min(h1, h2) / max(h1, h2)

        height_1 = self.heights[index1]
        height_2 = self.heights[index2]
        proposal_1 = self.text_proposals[index1]
        proposal_2 = self.text_proposals[index2]
        size_similarity_estimation = size_similarity(height_1, height_2)
        vertical_overlap = overlaps_v(height_1, height_2, proposal_1, proposal_2)

        return vertical_overlap >= 0.7 and size_similarity_estimation >= 0.7

    def build_graph(self, text_proposals, scores, im_size):
        self.text_proposals = text_proposals
        self.scores = scores
        self.im_size = im_size
        self.heights = text_proposals[:, 3] - text_proposals[:, 1] + 1

        boxes_table = [[] for _ in range(self.im_size[1])]
        for index, box in enumerate(text_proposals):
            boxes_table[int(box[0])].append(index)
        self.boxes_table = boxes_table

        graph = np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)

        for index, box in enumerate(text_proposals):
            successions = self.get_successions(index)
            if not successions:
                continue
            succession_index = successions[np.argmax(scores[successions])]
            if self.is_succession_node(index, succession_index):
                graph[index, succession_index] = True

        return Graph(graph)


class TextProposalConnector:
    def __init__(self):
        self.graph_builder = TextProposalGraphBuilder()

    def group_text_proposals(self, text_proposals, scores, im_size):
        graph = self.graph_builder.build_graph(text_proposals, scores, im_size)
        return graph.sub_graphs_connected()

    def get_text_lines(self, text_proposals, scores, im_size):
        def fit_y(x, y, x1, x2):
            if np.sum(x == x[0]) == np.size(x):
                return y[0], y[0]
            p = np.poly1d(np.polyfit(x, y, 1))
            return p(x1), p(x2)

        tp_groups = self.group_text_proposals(text_proposals, scores, im_size)
        text_lines = np.zeros((len(tp_groups), 5), np.float32)

        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes = text_proposals[list(tp_indices)]

            x0 = np.min(text_line_boxes[:, 0])
            x1 = np.max(text_line_boxes[:, 2])

            offset = (text_line_boxes[0, 2] - text_line_boxes[0, 0]) * 0.5

            lt_y, rt_y = fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0 + offset, x1 - offset)
            lb_y, rb_y = fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0 + offset, x1 - offset)
            score = scores[list(tp_indices)].sum() / float(len(tp_indices))

            text_lines[index, 0] = x0
            text_lines[index, 1] = min(lt_y, rt_y)
            text_lines[index, 2] = x1
            text_lines[index, 3] = max(lb_y, rb_y)
            text_lines[index, 4] = score

        text_lines = clip_boxes(text_lines, im_size)

        text_recs = np.zeros((len(text_lines), 9), np.float)
        index = 0
        for line in text_lines:
            xmin, ymin, xmax, ymax = line[0], line[1], line[2], line[3]
            text_recs[index, 0], text_recs[index, 1], text_recs[index, 2], text_recs[index, 3] = xmin, ymin, xmax, ymin
            text_recs[index, 4], text_recs[index, 5], text_recs[index, 6], text_recs[index, 7] = xmax, ymax, xmin, ymax
            text_recs[index, 8] = line[4]
            index = index + 1

        return text_recs


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
        if Polygon is None:
            raise ValueError("east_text_detection adapter requires shapely, please install it")

    def process(self, raw, identifiers, frame_meta):
        raw_outputs = self._extract_predictions(raw, frame_meta)
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
