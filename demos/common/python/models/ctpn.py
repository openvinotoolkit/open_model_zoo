"""
 Copyright (c) 2021 Intel Corporation

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

from .model import Model
from .utils import Detection, nms


class CTPN(Model):
    def __init__(self, ie, model_path, input_size, threshold=0.9):
        super().__init__(ie, model_path)

        self.image_blob_name = self.prepare_inputs()
        self.bboxes_blob_name, self.scores_blob_name = self.prepare_outputs()

        self.labels = ['Text']

        self.boxes_threshold = threshold
        self.nms_threshold = 0.5
        self.min_size = 8
        self.min_ratio = 0.5
        self.min_width = 32
        self.pre_nms_top_n = 1000
        self.post_nms_top_n = 500
        self.text_proposal_connector = TextProposalConnector()

        self.anchors = np.array([
            [0,   2,  15,  13],
            [0,   0,  15,  15],
            [0,  -4,  15,  19],
            [0,  -9,  15,  24],
            [0,  -16, 15,  31],
            [0,  -26, 15,  41],
            [0,  -41, 15,  56],
            [0,  -62, 15,  77],
            [0,  -91, 15, 106],
            [0, -134, 15, 149]
        ])

        self.h1, self.w1 = self.ctpn_keep_aspect_ratio(1200, 600, input_size[1], input_size[0])
        self.h2, self.w2 = self.ctpn_keep_aspect_ratio(600, 600, self.w1, self.h1)
        input_shape = {self.image_blob_name: (1, 3, self.h2, self.w2)}
        self.logger.info('Reshape net to {}'.format(input_shape))
        self.net.reshape(input_shape)

    def prepare_inputs(self):
        if len(self.net.input_info) != 1:
            raise RuntimeError("The CTPN topology supposes only 1 input layer")

        image_blob_name = next(iter(self.net.input_info))
        input_size = self.net.input_info[image_blob_name].input_data.shape

        if len(input_size) != 4 or input_size[1] != 3:
            raise RuntimeError("3-channel 4-dimensional model's input is expected")

        return image_blob_name

    def prepare_outputs(self):
        if len(self.net.outputs) != 2:
            raise RuntimeError("The CTPN topology supposes exactly 2 output layers")

        (boxes_name, boxes_data_repr), (scores_name, scores_data_repr) = self.net.outputs.items()

        if len(boxes_data_repr.shape) != 4 or len(scores_data_repr.shape) != 4:
            raise RuntimeError("Unexpected output blob shape. Only 4D output blobs are supported")

        if scores_data_repr.shape[1] == boxes_data_repr.shape[1] * 2:
            return scores_name, boxes_name
        if boxes_data_repr.shape[1] == scores_data_repr.shape[1] * 2:
            return boxes_name, scores_name
        raise RuntimeError("One of outputs must be two times larger than another for the CTPN topology")

    def preprocess(self, inputs):
        meta = {'original_shape': inputs.shape}
        scales = (self.w1 / inputs.shape[1], self.h1 / inputs.shape[0])

        if scales[0] < 1 and scales[1] < 1:
            meta['scales'] = [scales]
            inputs = cv2.resize(inputs, (self.w1, self.h1))
        if (self.h2 == 600 and self.w2 == 600 or
           (self.h1 != self.h2 or self.w1 != self.w2)):
            meta.setdefault('scales', []).append((self.w2 / inputs.shape[1],
                                                  self.h2 / inputs.shape[0]))
            inputs = cv2.resize(inputs, (self.w2, self.h2))

        inputs = inputs.transpose((2, 0, 1)) # Change data layout from HWC to CHW
        dict_inputs = {self.image_blob_name: inputs}
        return dict_inputs, meta

    def postprocess(self, outputs, meta):
        first_scales = meta['scales'].pop()
        boxes = outputs[self.bboxes_blob_name][0].transpose((1, 2, 0))
        scores = outputs[self.scores_blob_name][0].transpose((1, 2, 0))

        textsegs, scores = self.get_proposals(scores, boxes, meta['original_shape'])
        textsegs[:, 0::2] /= first_scales[0]
        textsegs[:, 1::2] /= first_scales[1]
        boxes = self.get_detections(textsegs, scores[:, np.newaxis], meta['original_shape'])
        if meta['scales']:
            second_scales = meta['scales'].pop()
            boxes[:, 0:8:2] /= second_scales[0]
            boxes[:, 1:8:2] /= second_scales[1]
        return [Detection(box[0], box[1], box[2], box[5], box[8], 0) for box in boxes]

    @staticmethod
    def ctpn_keep_aspect_ratio(dst_width, dst_height, image_width, image_height):
        scale = min(dst_height, dst_width)
        max_scale = max(dst_height, dst_width)
        im_min_size = min(image_width, image_height)
        im_max_size = max(image_width, image_height)
        im_scale = float(scale) / float(im_min_size)
        if np.round(im_scale * im_max_size) > max_scale:
            im_scale = float(max_scale) / float(im_max_size)
        new_h = np.round(image_height * im_scale)
        new_w = np.round(image_width * im_scale)

        return int(new_h), int(new_w)

    def get_proposals(self, rpn_cls_prob_reshape, bbox_deltas, image_size, _feat_stride=16):
        """
        Parameters
        rpn_cls_prob_reshape: (H , W , Ax2), probabilities for predicted regions
        bbox_deltas: (H , W , Ax4), predicted regions
        image_size: a list of [image_height, image_width]
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

        # clip predicted boxes to image
        proposals[:, :4].clip(min=0, max=(image_size[1] - 1, image_size[0] - 1, image_size[1] - 1, image_size[0] - 1),
                              out=proposals[:, :4])
        # sort all (proposal, score) pairs by score from highest to lowest
        order = scores.ravel().argsort()[::-1]
        if self.pre_nms_top_n > 0:
            order = order[:self.pre_nms_top_n]
        proposals, scores = proposals[order, :], scores[order]

        # apply nms
        keep = nms(proposals[:, 0], proposals[:, 1], proposals[:, 2], proposals[:, 3], scores.reshape(-1),
                   self.nms_threshold, include_boundaries=True)
        if self.post_nms_top_n > 0:
            keep = keep[:self.post_nms_top_n]
        proposals, scores = proposals[keep, :], scores[keep]
        return proposals, scores

    def get_detections(self, text_proposals, scores, size):
        keep_inds = np.where(scores > 0.7)[0]
        text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]

        sorted_indices = np.argsort(scores.ravel())[::-1]
        text_proposals, scores = text_proposals[sorted_indices], scores[sorted_indices]

        text_recs = self.text_proposal_connector.get_text_lines(text_proposals, scores, size)

        heights = (abs(text_recs[:, 5] - text_recs[:, 1]) + abs(text_recs[:, 7] - text_recs[:, 3])) / 2.0 + 1
        widths = (abs(text_recs[:, 2] - text_recs[:, 0]) + abs(text_recs[:, 6] - text_recs[:, 4])) / 2.0 + 1
        scores = text_recs[:, 8]
        keep_inds = np.where((widths / heights > self.min_ratio) & (scores > self.boxes_threshold) &
                             (widths > self.min_width))[0]

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

    def group_text_proposals(self, text_proposals, scores, image_size):
        graph = self.graph_builder.build_graph(text_proposals, scores, image_size)
        return graph.sub_graphs_connected()

    def get_text_lines(self, text_proposals, scores, image_size):
        def fit_y(x, y, x1, x2):
            if np.sum(x == x[0]) == np.size(x):
                return y[0], y[0]
            p = np.poly1d(np.polyfit(x, y, 1))
            return p(x1), p(x2)

        tp_groups = self.group_text_proposals(text_proposals, scores, image_size)

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

        text_lines[:, :4].clip(min=0, max=(image_size[1] - 1, image_size[0] - 1, image_size[1] - 1, image_size[0] - 1),
                               out=text_lines[:, :4])

        text_recs = np.zeros((len(text_lines), 9), np.float)
        for index, line in enumerate(text_lines):
            xmin, ymin, xmax, ymax = line[0], line[1], line[2], line[3]
            text_recs[index, 0], text_recs[index, 1], text_recs[index, 2], text_recs[index, 3] = xmin, ymin, xmax, ymin
            text_recs[index, 4], text_recs[index, 5], text_recs[index, 6], text_recs[index, 7] = xmax, ymax, xmin, ymax
            text_recs[index, 8] = line[4]

        return text_recs
