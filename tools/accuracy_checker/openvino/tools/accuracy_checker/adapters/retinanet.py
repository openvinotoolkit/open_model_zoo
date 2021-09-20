"""
Copyright (c) 2018-2021 Intel Corporation
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

from collections import OrderedDict
import numpy as np

from .adapter import Adapter
from ..config import ListField, StringField, NumberField
from ..representation import DetectionPrediction
from ..postprocessor import NMS


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
            'post_nms_top_k': NumberField(
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

            boxes = np.concatenate([
                pred_ctr - 0.5 * pred_wh,
                pred_ctr + 0.5 * pred_wh - 1,
            ], axis=1)

            M = np.array([*size, *size]) * stride - 1
            return np.clip(boxes, 0, M)

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
            'class_out': StringField(description="output with classes probabilities")
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


class RetinaNetTF2(Adapter):
    __provider__ = 'retinanet_tf2'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'boxes_outputs': ListField(description='boxes localization output', value_type=str),
            'class_outputs': ListField(description="output with classes probabilities"),
            'min_level': NumberField(optional=True, value_type=int, default=3, description='min pyramid level'),
            'max_level': NumberField(optional=True, value_type=int, default=7, description='max pyramid level'),
            'aspect_ratios': ListField(
                value_type=float, optional=True, default=[1, 2, 0.5], description='aspect ratio levels'
            ),
            'num_scales': NumberField(
                optional=True, default=3, value_type=int, min_value=1, description='number anchor scales'),
            'anchor_size': NumberField(optional=True, default=4, description='anchor box size'),
            'total_size': NumberField(
                optional=True, default=100, value_type=int, min_value=1, description='final number of boxes'
            ),
            'pre_nms_top_k': NumberField(
                optional=True, value_type=int, min_value=1, default=5000,
                description='number of keep top by score boxes before nms'),
            'score_threshold': NumberField(
                value_type=float, min_value=0, max_value=1, default=0.05, description='scores threshold'
            ),
            'nms_threshold': NumberField(
                value_type=float, min_value=0, max_value=1, default=0.5, description='nms threshold'
            )

        })
        return params

    def configure(self):
        self.loc_out = self.get_value_from_config('boxes_outputs')
        self.cls_out = self.get_value_from_config('class_outputs')
        self.min_level = self.get_value_from_config('min_level')
        self.max_level = self.get_value_from_config('max_level')
        self.aspect_ratios = self.get_value_from_config('aspect_ratios')
        self.anchor_size = self.get_value_from_config('anchor_size')
        self.num_scales = self.get_value_from_config('num_scales')
        self.max_total_size = self.get_value_from_config('total_size')
        self.nms_iou_threshold = self.get_value_from_config('nms_threshold')
        self.score_threshold = self.get_value_from_config('score_threshold')
        self.pre_nms_num_boxes = self.get_value_from_config('pre_nms_top_k')

    def _generate_anchor_boxes(self, image_size):
        boxes_all = []
        for level in range(self.min_level, self.max_level + 1):
            boxes_l = []
            for scale in range(self.num_scales):
                for aspect_ratio in self.aspect_ratios:
                    stride = 2 ** level
                    intermediate_scale = 2 ** (scale / float(self.num_scales))
                    base_anchor_size = self.anchor_size * stride * intermediate_scale
                    aspect_x = aspect_ratio ** 0.5
                    aspect_y = aspect_ratio ** -0.5
                    half_anchor_size_x = base_anchor_size * aspect_x / 2.0
                    half_anchor_size_y = base_anchor_size * aspect_y / 2.0
                    x = np.arange(stride / 2, image_size[1], stride)
                    y = np.arange(stride / 2, image_size[0], stride)
                    xv, yv = np.meshgrid(x, y)
                    xv = np.reshape(xv, -1)
                    yv = np.reshape(yv, -1)
                    boxes = np.stack([
                        yv - half_anchor_size_y, xv - half_anchor_size_x,
                        yv + half_anchor_size_y, xv + half_anchor_size_x
                    ], axis=1)
                    boxes_l.append(boxes)
            boxes_l = np.stack(boxes_l, axis=1)
            boxes_l = np.reshape(boxes_l, [-1, 4])
            boxes_all.append(boxes_l)

        def unpack_labels(labels):
            unpacked_labels = OrderedDict()
            count = 0
            for level in range(self.min_level, self.max_level + 1):
                feat_size_y = int(image_size[0] / 2 ** level)
                feat_size_x = int(image_size[1] / 2 ** level)
                steps = feat_size_y * feat_size_x * self.num_scales * len(self.aspect_ratios)
                unpacked_labels[level] = np.reshape(labels[count:count + steps],
                                                    [feat_size_y, feat_size_x, -1])
                count += steps
            return unpacked_labels

        return unpack_labels(np.concatenate(boxes_all, axis=0))

    def prepare_boxes_and_classes(self, raw, batch_id):
        boxes_outs, classes_outs = [], []
        for boxes_out, cls_out in zip(self.loc_out, self.cls_out):
            boxes_outs.append(np.transpose(raw[boxes_out][batch_id], (1, 2, 0)))
            classes_outs.append(np.transpose(raw[cls_out][batch_id], (1, 2, 0)))
        return boxes_outs, classes_outs

    def process(self, raw, identifiers, frame_meta):
        raw_outputs = self._extract_predictions(raw, frame_meta)
        result = []
        for batch_id, (identifier, meta) in enumerate(zip(identifiers, frame_meta)):
            boxes_out, classes_out = self.prepare_boxes_and_classes(raw_outputs, batch_id)
            input_shape = [shape for shape in meta['input_shape'].values() if len(shape) == 4]
            input_shape = input_shape[0]
            image_size = input_shape[2:] if input_shape[1] == 3 else input_shape[1:3]
            boxes, scores, labels = self.process_single(boxes_out, classes_out, image_size)
            if np.size(boxes):
                x_mins, y_mins, x_maxs, y_maxs = boxes.T
                x_mins /= image_size[1]
                y_mins /= image_size[0]
                x_maxs /= image_size[1]
                y_maxs /= image_size[0]
            else:
                x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
            result.append(
                DetectionPrediction(
                    identifier, labels, scores,
                    x_mins, y_mins, x_maxs, y_maxs
                ))
        return result

    def process_single(self, box_outputs, class_outputs, image_size):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        boxes = []
        scores = []
        anchor_boxes = self._generate_anchor_boxes(image_size)
        for i in range(self.min_level, self.max_level + 1):
            box_outputs_i_shape = np.shape(box_outputs[i - self.min_level])
            num_anchors_per_locations = box_outputs_i_shape[-1] // 4
            num_classes = np.shape(class_outputs[i - self.min_level])[-1] // num_anchors_per_locations

            scores_i = sigmoid(np.reshape(class_outputs[i - self.min_level], [-1, num_classes]))
            scores_i = scores_i[:, 1:]

            anchor_boxes_i = np.reshape(anchor_boxes[i], [-1, 4])
            box_outputs_i = np.reshape(box_outputs[i - self.min_level], [-1, 4])
            boxes_i = self.decode_boxes(box_outputs_i, anchor_boxes_i)
            boxes_i[:, ::2] = np.clip(boxes_i[:, ::2], a_min=0, a_max=image_size[1] - 1)
            boxes_i[:, 1::2] = np.clip(boxes_i[:, 1::2], a_min=0, a_max=image_size[0] - 1)

            boxes.append(boxes_i)
            scores.append(scores_i)
        boxes = np.concatenate(boxes, axis=0)
        scores = np.concatenate(scores, axis=0)

        nmsed_boxes, nmsed_scores, nmsed_classes = self._generate_detections(
            np.expand_dims(boxes, axis=1), scores,
            self.max_total_size, self.nms_iou_threshold, self.score_threshold, self.pre_nms_num_boxes
        )

        return nmsed_boxes, nmsed_scores, nmsed_classes

    @staticmethod
    def decode_boxes(encoded_boxes, anchors):
        BBOX_XFORM_CLIP = np.log(1000. / 16.)
        dy = encoded_boxes[..., 0:1]
        dx = encoded_boxes[..., 1:2]
        dh = encoded_boxes[..., 2:3]
        dw = encoded_boxes[..., 3:4]
        dh = np.minimum(dh, BBOX_XFORM_CLIP)
        dw = np.minimum(dw, BBOX_XFORM_CLIP)

        anchor_ymin = anchors[..., 0:1]
        anchor_xmin = anchors[..., 1:2]
        anchor_ymax = anchors[..., 2:3]
        anchor_xmax = anchors[..., 3:4]
        anchor_h = anchor_ymax - anchor_ymin + 1.0
        anchor_w = anchor_xmax - anchor_xmin + 1.0
        anchor_yc = anchor_ymin + 0.5 * anchor_h
        anchor_xc = anchor_xmin + 0.5 * anchor_w

        decoded_boxes_yc = dy * anchor_h + anchor_yc
        decoded_boxes_xc = dx * anchor_w + anchor_xc
        decoded_boxes_h = np.exp(dh) * anchor_h
        decoded_boxes_w = np.exp(dw) * anchor_w

        decoded_boxes_ymin = decoded_boxes_yc - 0.5 * decoded_boxes_h
        decoded_boxes_xmin = decoded_boxes_xc - 0.5 * decoded_boxes_w
        decoded_boxes_ymax = decoded_boxes_ymin + decoded_boxes_h - 1.0
        decoded_boxes_xmax = decoded_boxes_xmin + decoded_boxes_w - 1.0

        decoded_boxes = np.concatenate([
            decoded_boxes_xmin, decoded_boxes_ymin, decoded_boxes_xmax,
            decoded_boxes_ymax
        ], axis=-1)
        return decoded_boxes

    @staticmethod
    def _generate_detections(boxes,
                             scores,
                             max_total_size=100,
                             nms_iou_threshold=0.5,
                             score_threshold=0.05,
                             pre_nms_num_boxes=5000):

        def _select_top_k_scores(scores_in, pre_nms_num_detections):
            num_anchors, num_class = scores_in.shape
            scores_trans = np.transpose(scores_in, [1, 0])
            scores_trans = np.reshape(scores_trans, [-1, num_anchors])

            indices_ = np.argsort(-scores_trans)
            top_k_scores = -1 * np.sort(-scores_trans)[:, :pre_nms_num_detections]
            top_k_indices = indices_[:, :pre_nms_num_detections]

            top_k_scores = np.reshape(top_k_scores,
                                      [num_class, pre_nms_num_detections])
            top_k_indices = np.reshape(top_k_indices,
                                       [num_class, pre_nms_num_detections])

            return np.transpose(top_k_scores,
                                [1, 0]), np.transpose(top_k_indices, [1, 0])

        nmsed_boxes = []
        nmsed_classes = []
        nmsed_scores = []
        _, num_classes_for_box, _ = boxes.shape
        total_anchors, num_classes = scores.shape
        scores, indices = _select_top_k_scores(
            scores, min(total_anchors, pre_nms_num_boxes))
        for i in range(num_classes):
            boxes_i = boxes[:, min(num_classes_for_box - 1, i), :]
            scores_i = scores[:, i]
            boxes_i = boxes_i[indices[:, i], :]

            filtered_scores = scores_i > score_threshold
            boxes_i = boxes_i[filtered_scores]
            scores_i = scores_i[filtered_scores]
            if not np.size(scores_i):
                continue

            keep = NMS.nms(*boxes_i.T, scores_i, nms_iou_threshold)
            if len(keep) > max_total_size:
                keep = keep[:max_total_size]
            nms_boxes = boxes_i[keep]
            nms_scores = scores_i[keep]
            nmsed_classes_i = np.full(len(nms_scores), i+1)
            nmsed_boxes.append(nms_boxes)
            nmsed_scores.append(nms_scores)
            nmsed_classes.append(nmsed_classes_i)
        if np.size(nmsed_scores):
            nmsed_boxes = np.concatenate(nmsed_boxes, axis=0)
            nmsed_scores = np.concatenate(nmsed_scores, axis=0)
            nmsed_classes = np.concatenate(nmsed_classes, axis=0)
            sorted_order = np.argsort(nmsed_scores)[::-1]
            if sorted_order.size > max_total_size:
                sorted_order = sorted_order[:max_total_size]
            nmsed_scores = nmsed_scores[sorted_order]
            nmsed_boxes = nmsed_boxes[sorted_order, :]
            nmsed_classes = nmsed_classes[sorted_order]
        return nmsed_boxes, nmsed_scores, nmsed_classes
