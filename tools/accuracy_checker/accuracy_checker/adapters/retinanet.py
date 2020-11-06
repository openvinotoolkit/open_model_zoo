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

import numpy as np

from .adapter import Adapter
from ..config import ListField, StringField, NumberField
from ..representation import DetectionPrediction


class MultiOutRetinaNet(Adapter):
    __provider__ = 'retinanet_multihead'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'boxes_outputs': ListField(description='boxes localization outputs'),
            'class_outputs': ListField(description="outputs with classes probabilities"),
            'ratios': ListField(
                description='ratio for anchors generation', optional=True, default=[1.0, 2.0, 0.5], value_type=float
            ),
            'pre_nms_top_k': NumberField(
                description='pre nms keep top k boxes', value_type=int, optional=True, default=1000
            ),
            'post_nms_top_k':  NumberField(
                description='post nms keep top k boxes', value_type=int, optional=True, default=100
            ),
            'min_conf': NumberField(
                description='min score for detection filtering', value_type=float, optional=True, default=0.05
            ),
            'nms_threshold': NumberField(
                description='overlap threshold for nms', value_type=float, optional=True, default=0.5
            )
        })
        return params

    def configure(self):
        self.ratios = self.get_value_from_config('ratios')
        self.scales = [4 * 2 ** (i / 3) for i in range(3)]

        self.boxes_outs = self.get_value_from_config('boxes_outputs')
        self.class_outs = self.get_value_from_config('class_outputs')
        self.anchors = {}
        assert len(self.boxes_outs) == len(self.class_outs), 'the number of boxes and classes heads should be equal'
        self.pre_nms_top_k = self.get_value_from_config('pre_nms_top_k')
        self.post_nms_top_k = self.get_value_from_config('post_nms_top_k')
        self.min_conf = self.get_value_from_config('min_conf')
        self.nms_threshold = self.get_value_from_config('nms_threshold')

    def decode_boxes(self, raw_outputs, input_shape):
        def generate_anchors(stride, ratio_vals, scales_vals):
            scales = np.tile(np.array(scales_vals), (len(ratio_vals), 1))
            scales = np.transpose(scales, (0, 1)).reshape((-1, 1))
            ratios = ratio_vals * len(scales_vals)

            wh = np.tile(np.array([stride]), (len(ratios), 2))
            ws = np.sqrt(wh[:, 0] * wh[:, 1] / ratios)
            dwh = np.stack([ws, ws * ratios], axis=1)
            xy1 = 0.5 * (wh - dwh * scales)
            xy2 = 0.5 * (wh + dwh * scales)
            return np.concatenate([xy1, xy2], axis=1)

        cls_heads = [raw_outputs[cls_out] for cls_out in self.class_outs]
        box_heads = [raw_outputs[box_out] for box_out in self.boxes_outs]
        decoded = []
        for cls_head, box_head in zip(cls_heads, box_heads):
            # Generate level's anchors
            stride = input_shape[-1] // cls_head.shape[-1]
            if stride not in self.anchors:
                self.anchors[stride] = generate_anchors(stride, self.ratios, self.scales)

            # Decode and filter boxes
            decoded.append(
                self.decode(cls_head, box_head, stride, self.min_conf, self.pre_nms_top_k,
                            anchors=self.anchors[stride])
            )

        # Perform non-maximum suppression
        decoded = [np.concatenate(tensors, 1) for tensors in zip(*decoded)]
        return self.nms(*decoded, nms=self.nms_threshold, ndetections=self.post_nms_top_k)

    @staticmethod
    def decode(all_cls_head, all_box_head, stride=1, threshold=0.05, top_n=1000, anchors=None):
        def delta2box(deltas, anchors, size, stride):
            'Convert deltas from anchors to boxes'

            anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
            ctr = anchors[:, :2] + 0.5 * anchors_wh
            pred_ctr = deltas[:, :2] * anchors_wh + ctr
            pred_wh = np.exp(deltas[:, 2:]) * anchors_wh

            m = np.zeros([2])
            M = np.array([size]) * stride - 1
            clamp = lambda t: np.maximum(m, np.minimum(t, M))
            return np.concatenate([
                clamp(pred_ctr - 0.5 * pred_wh),
                clamp(pred_ctr + 0.5 * pred_wh - 1)
            ], 1)

        num_boxes = 4
        num_anchors = anchors.shape[0] if anchors is not None else 1
        num_classes = all_cls_head.shape[1] // num_anchors
        height, width = all_cls_head.shape[-2:]

        batch_size = all_cls_head.shape[0]
        out_scores = np.zeros((batch_size, top_n))
        out_boxes = np.zeros((batch_size, top_n, num_boxes))
        out_classes = np.zeros((batch_size, top_n))

        # Per item in batch
        for batch in range(batch_size):
            cls_head = all_cls_head[batch, :, :, :].reshape(-1)
            box_head = all_box_head[batch, :, :, :].reshape(-1, num_boxes)

            # Keep scores over threshold
            keep = np.nonzero(cls_head >= threshold)[0]
            if np.size(keep) == 0:
                continue

            # Gather top elements
            scores = cls_head[keep]
            indices = np.argsort(scores)[::-1]
            indices = indices[:min(top_n, keep.size)]
            scores = scores[indices]
            indices = keep[indices]
            classes = (indices / width / height) % num_classes
            classes = classes.astype(int)

            # Infer kept bboxes
            x = indices % width
            y = (indices // width) % height
            a = indices // num_classes // height // width
            box_head = box_head.reshape(num_anchors, num_boxes, height, width)
            boxes = box_head[a, :, y, x]

            if anchors is not None:
                grid = np.stack([x, y, x, y], 1) * stride + anchors[a, :]
                boxes = delta2box(boxes, grid, [width, height], stride)

            out_scores[batch, :scores.shape[0]] = scores
            out_boxes[batch, :boxes.shape[0], :] = boxes
            out_classes[batch, :classes.shape[0]] = classes

        return out_scores, out_boxes, out_classes

    @staticmethod
    def nms(all_scores, all_boxes, all_classes, nms=0.5, ndetections=100):
        'Non Maximum Suppression'
        batch_size = all_scores.shape[0]
        out_scores = np.zeros((batch_size, ndetections))
        out_boxes = np.zeros((batch_size, ndetections, 4))
        out_classes = np.zeros((batch_size, ndetections))

        # Per item in batch
        for batch in range(batch_size):
            # Discard null scores
            keep = (all_scores[batch, :].reshape(-1) > 0).nonzero()
            scores = all_scores[batch, keep].reshape(-1)
            boxes = all_boxes[batch, keep, :].reshape(-1, 4)
            classes = all_classes[batch, keep].reshape(-1)

            if scores.size == 0:
                continue

            # Sort boxes
            indices = np.argsort(scores)[::-1]
            boxes, classes, scores = boxes[indices], classes[indices], scores[indices]
            areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1).reshape(-1)
            keep = np.ones(len(scores))

            for i in range(ndetections):
                if i >= keep.nonzero()[0].size or i >= scores.size:
                    i -= 1
                    break

                # Find overlapping boxes with lower score
                xy1 = np.maximum(boxes[:, :2], boxes[i, :2])
                xy2 = np.minimum(boxes[:, 2:], boxes[i, 2:])
                inter = np.prod((xy2 - xy1 + 1).clip(0), 1)
                criterion = ((scores > scores[i]) |
                             (inter / (areas + areas[i] - inter) <= nms) |
                             (classes != classes[i]))
                criterion[i] = 1

                # Only keep relevant boxes
                scores = scores[criterion.nonzero()].reshape(-1)
                boxes = boxes[criterion.nonzero(), :].reshape(-1, 4)
                classes = classes[criterion.nonzero()].reshape(-1)
                areas = areas[criterion.nonzero()].reshape(-1)
                keep[(~criterion).nonzero()] = 0

            if i >= scores.size:
                i = scores.size - 2
            out_scores[batch, :i + 1] = scores[:i + 1]
            out_boxes[batch, :i + 1, :] = boxes[:i + 1, :]
            out_classes[batch, :i + 1] = classes[:i + 1]

        return out_scores, out_boxes, out_classes

    def process(self, raw, identifiers, frame_meta):
        raw_outputs = self._extract_predictions(raw, frame_meta)
        input_shape_dict = frame_meta[0].get('input_shape', {'data': (1, 3, 480, 640)})
        input_shape = next(iter(input_shape_dict.values()))
        out_scores, out_boxes, out_classes = self.decode_boxes(raw_outputs, input_shape)
        result = []
        for identifier, boxes, scores, labels in zip(identifiers, out_boxes, out_scores, out_classes):
            non_empty = (scores > 0).nonzero()[0]
            result.append(DetectionPrediction(identifier, labels[non_empty], scores[non_empty], *boxes[non_empty].T))
        return result


class RetinaNetAdapter(Adapter):
    __provider__ = 'retinanet'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'loc_out': StringField(description='boxes localization output'),
            'class_out':  StringField(description="output with classes probabilities")
        })
        return params

    def configure(self):
        self.loc_out = self.get_value_from_config('loc_out')
        self.cls_out = self.get_value_from_config('class_out')
        self.pyramid_levels = [3, 4, 5, 6, 7]
        self.strides = [2 ** x for x in self.pyramid_levels]
        self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        self.ratios = np.array([0.5, 1, 2])
        self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
        self.std = np.array([0.1, 0.1, 0.2, 0.2])

    def process(self, raw, identifiers, frame_meta):
        raw_outputs = self._extract_predictions(raw, frame_meta)
        results = []
        for identifier, loc_pred, cls_pred, meta in zip(
                identifiers, raw_outputs[self.loc_out], raw_outputs[self.cls_out], frame_meta
        ):
            _, _, h, w = next(iter(meta.get('input_shape', {'data': (1, 3, 800, 800)}).values()))
            anchors = self.create_anchors([w, h])
            transformed_anchors = self.regress_boxes(anchors, loc_pred)
            labels, scores = np.argmax(cls_pred, axis=1), np.max(cls_pred, axis=1)
            scores_mask = np.reshape(scores > 0.05, -1)
            transformed_anchors = transformed_anchors[scores_mask, :]
            x_mins, y_mins, x_maxs, y_maxs = transformed_anchors.T
            results.append(DetectionPrediction(
                identifier, labels[scores_mask], scores[scores_mask], x_mins / w, y_mins / h, x_maxs / w, y_maxs / h
            ))

        return results

    def create_anchors(self, input_shape):
        def _generate_anchors(base_size=16):
            """
            Generate anchor (reference) windows by enumerating aspect ratios X
            scales w.r.t. a reference window.
            """
            num_anchors = len(self.ratios) * len(self.scales)
            # initialize output anchors
            anchors = np.zeros((num_anchors, 4))
            # scale base_size
            anchors[:, 2:] = base_size * np.tile(self.scales, (2, len(self.ratios))).T
            # compute areas of anchors
            areas = anchors[:, 2] * anchors[:, 3]
            # correct for ratios
            anchors[:, 2] = np.sqrt(areas / np.repeat(self.ratios, len(self.scales)))
            anchors[:, 3] = anchors[:, 2] * np.repeat(self.ratios, len(self.scales))
            # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
            anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
            anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

            return anchors

        def _shift(shape, stride, anchors):
            shift_x = (np.arange(0, shape[1]) + 0.5) * stride
            shift_y = (np.arange(0, shape[0]) + 0.5) * stride
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)

            shifts = np.vstack((
                shift_x.ravel(), shift_y.ravel(),
                shift_x.ravel(), shift_y.ravel()
            )).transpose()
            a = anchors.shape[0]
            k = shifts.shape[0]
            all_anchors = (anchors.reshape((1, a, 4)) + shifts.reshape((1, k, 4)).transpose((1, 0, 2)))
            all_anchors = all_anchors.reshape((k * a, 4))

            return all_anchors

        image_shapes = [(np.array(input_shape) + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]
        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4)).astype(np.float32)
        for idx, _ in enumerate(self.pyramid_levels):
            anchors = _generate_anchors(base_size=self.sizes[idx])
            shifted_anchors = _shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

        return all_anchors

    def regress_boxes(self, boxes, deltas):
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        dx = deltas[:, 0] * self.std[0]
        dy = deltas[:, 1] * self.std[1]
        dw = deltas[:, 2] * self.std[2]
        dh = deltas[:, 3] * self.std[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = np.exp(dw) * widths
        pred_h = np.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = np.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], axis=1)

        return pred_boxes
