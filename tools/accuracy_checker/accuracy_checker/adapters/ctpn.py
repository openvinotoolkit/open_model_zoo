import numpy as np

from .adapter import Adapter
from ..config import StringField, NumberField
from ..postprocessor import NMS
from ..representation import TextDetectionPrediction
from ..utils import UnsupportedPackage

try:
    from shapely.geometry import Polygon
except ImportError as import_error:
    Polygon = UnsupportedPackage("shapely", import_error.msg)


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
        text_proposals_width = self.get_value_from_config('text_proposals_width')
        min_num_proposals = self.get_value_from_config('min_num_proposals')
        self.min_width = min_num_proposals * text_proposals_width
        if isinstance(Polygon, UnsupportedPackage):
            Polygon.raise_error(self.__provider__)
        self.text_proposal_connector = TextProposalConnector()
        self.anchors = np.array([
            [0, 2, 15, 13],
            [0, 0, 15, 15],
            [0, -4, 15, 19],
            [0, -9, 15, 24],
            [0, -16, 15, 31],
            [0, -26, 15, 41],
            [0, -41, 15, 56],
            [0, -62, 15, 77],
            [0, -91, 15, 106],
            [0, -134, 15, 149]
        ])
        self.outputs_verified = False

    def select_output_blob(self, outputs):
        self.cls_prob_out = self.check_output_name(self.cls_prob_out, outputs)
        self.bbox_pred_out = self.check_output_name(self.bbox_pred_out, outputs)
        self.outputs_verified = True

    def process(self, raw, identifiers, frame_meta):
        if not self.outputs_verified:
            self.select_output_blob(raw)
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
            textsegs, scores = self.get_proposals(cls_prob, bbox_pred, im_info)
            textsegs[:, 0::2] /= scale_x
            textsegs[:, 1::2] /= scale_y
            boxes = self.get_detections(textsegs, scores[:, np.newaxis],
                                        [meta['original_height'], meta['original_width']])
            boxes = boxes[:, :8]
            geom_operations = meta['geometric_operations']
            resize_op = [geom_operation for geom_operation in geom_operations if geom_operation.type == 'resize']
            if len(resize_op) >= 2:
                scale_x, scale_y = resize_op[0].parameters['scale_y'], resize_op[0].parameters['scale_x']
                boxes[:, 0::2] /= scale_x
                boxes[:, 1::2] /= scale_y
            rects = [box.reshape(4, 2) for box in boxes]
            result.append(TextDetectionPrediction(identifier, np.array(rects)))

        return result

    def get_proposals(self, rpn_cls_prob_reshape, bbox_deltas, im_info, _feat_stride=16):
        """
        Parameters
        rpn_cls_prob_reshape: (H , W , Ax2) probabilities for predicted regions
        bbox_deltas: (H , W , Ax4), predicted regions
        im_info: a list of [image_height, image_width, scale_ratios]
        _feat_stride: the downsampling ratio of feature map to the original input image
        Algorithm:
        for each (H, W) location i
        generate A anchor boxes centered on location i
        apply predicted bbox deltas at location i to each of the A anchors
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

        _anchors = self.anchors.copy()
        _num_anchors = _anchors.shape[0]
        height, width = rpn_cls_prob_reshape.shape[:2]
        scores = np.reshape(
            np.reshape(rpn_cls_prob_reshape, [height, width, _num_anchors, 2])[:, :, :, 1],
            [height, width, _num_anchors]
        )

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
        proposals[:, :4].clip(min=0, max=(im_info[1] - 1, im_info[0] - 1, im_info[1] - 1, im_info[0] - 1),
                              out=proposals[:, :4])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = _filter_boxes(proposals, self.min_size * im_info[2])
        proposals, scores = proposals[keep, :], scores[keep]
        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if self.pre_nms_top_n > 0:
            order = order[:self.pre_nms_top_n]
        proposals, scores = proposals[order, :], scores[order]
        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        keep = NMS.nms(
            proposals[:, 0], proposals[:, 1], proposals[:, 2], proposals[:, 3], scores.reshape(-1), self.nms_threshold
        )
        if self.post_nms_top_n > 0:
            keep = keep[:self.post_nms_top_n]
        proposals, scores = proposals[keep, :], scores[keep]
        return proposals, scores

    def get_detections(self, text_proposals, scores, size):
        keep_inds = np.where(scores > 0.7)[0]
        text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]

        sorted_indices = np.argsort(scores.ravel())[::-1]
        text_proposals, scores = text_proposals[sorted_indices], scores[sorted_indices]
        x_mins, y_mins = text_proposals[:, 0], text_proposals[:, 1]
        x_maxs, y_maxs = text_proposals[:, 2], text_proposals[:, 3]

        keep_inds = NMS.nms(x_mins, y_mins, x_maxs, y_maxs, scores.reshape(-1), 0.2)
        text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]

        text_recs = self.text_proposal_connector.get_text_lines(text_proposals, scores, size)

        heights = (abs(text_recs[:, 5] - text_recs[:, 1]) + abs(text_recs[:, 7] - text_recs[:, 3])) / 2.0 + 1
        widths = (abs(text_recs[:, 2] - text_recs[:, 0]) + abs(text_recs[:, 6] - text_recs[:, 4])) / 2.0 + 1
        scores = text_recs[:, 8]
        keep_inds = np.where(
            (widths / heights > self.min_ratio) & (scores > self.line_min_score) &
            (widths > self.min_width)
        )[0]

        return text_recs[keep_inds]

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
        return self.scores[index] >= np.max(self.scores[precursors])

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

        graph = np.zeros((text_proposals.shape[0], text_proposals.shape[0]), bool)

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

        text_lines[:, :4].clip(min=0, max=(im_size[1] - 1, im_size[0] - 1, im_size[1] - 1, im_size[0] - 1),
                               out=text_lines[:, :4])

        text_recs = np.zeros((len(text_lines), 9), float)
        for index, line in enumerate(text_lines):
            xmin, ymin, xmax, ymax = line[0], line[1], line[2], line[3]
            text_recs[index, 0], text_recs[index, 1], text_recs[index, 2], text_recs[index, 3] = xmin, ymin, xmax, ymin
            text_recs[index, 4], text_recs[index, 5], text_recs[index, 6], text_recs[index, 7] = xmax, ymax, xmin, ymax
            text_recs[index, 8] = line[4]

        return text_recs
