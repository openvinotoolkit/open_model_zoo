import numpy as np
from ..adapters import Adapter
from ..config import ListField
from ..postprocessor import NMS
from ..representation import (
    DetectionPrediction,
    FacialLandmarksPrediction,
    ContainerPrediction,
    AttributeDetectionPrediction
)


class RetinaFaceAdapter(Adapter):
    __provider__ = 'retinaface'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update(
            {
                'bboxes_outputs': ListField(),
                'scores_outputs': ListField(),
                'landmarks_outputs': ListField(optional=True),
                'type_scores_outputs': ListField(optional=True)
            }
        )
        return params

    def configure(self):
        self.bboxes_output = self.get_value_from_config('bboxes_outputs')
        self.scores_output = self.get_value_from_config('scores_outputs')
        self.landmarks_output = self.get_value_from_config('landmarks_outputs') or []
        self.type_scores_output = self.get_value_from_config('type_scores_outputs') or []
        _ratio = (1.,)
        self.anchor_cfg = {
            32: {'SCALES': (32, 16), 'BASE_SIZE': 16, 'RATIOS': _ratio},
            16: {'SCALES': (8, 4), 'BASE_SIZE': 16, 'RATIOS': _ratio},
            8: {'SCALES': (2, 1), 'BASE_SIZE': 16, 'RATIOS': _ratio}
        }
        self._features_stride_fpn = [32, 16, 8]
        self._anchors_fpn = dict(zip(self._features_stride_fpn, self.generate_anchors_fpn(cfg=self.anchor_cfg)))
        self._num_anchors = dict(zip(
            self._features_stride_fpn, [anchors.shape[0] for anchors in self._anchors_fpn.values()]
        ))
        if self.type_scores_output:
            self.landmark_std = 0.2
        else:
            self.landmark_std = 1.0

    def process(self, raw, identifiers=None, frame_meta=None):
        raw_predictions = self._extract_predictions(raw, frame_meta)
        results = []
        for batch_id, (identifier, meta) in enumerate(zip(identifiers, frame_meta)):
            proposals_list = []
            scores_list = []
            landmarks_list = []
            mask_scores_list = []
            x_scale, y_scale = meta['scale_x'], meta['scale_y']
            for _idx, s in enumerate(self._features_stride_fpn):
                anchor_num = self._num_anchors[s]
                scores = self._get_scores(raw_predictions[self.scores_output[_idx]][batch_id], anchor_num)
                bbox_deltas = raw_predictions[self.bboxes_output[_idx]][batch_id]
                height, width = bbox_deltas.shape[1], bbox_deltas.shape[2]
                anchors_fpn = self._anchors_fpn[s]
                anchors = self.anchors_plane(height, width, int(s), anchors_fpn)
                anchors = anchors.reshape((height * width * anchor_num, 4))
                proposals = self._get_proposals(bbox_deltas, anchor_num, anchors)
                x_mins, y_mins, x_maxs, y_maxs = proposals.T
                keep = NMS.nms(x_mins, y_mins, x_maxs, y_maxs, scores, 0.5, False)
                proposals_list.extend(proposals[keep])
                scores_list.extend(scores[keep])
                if self.type_scores_output:
                    mask_scores_list.extend(self._get_mask_scores(
                        raw_predictions[self.type_scores_output[_idx]][batch_id], anchor_num)[keep])
                if self.landmarks_output:
                    landmarks = self._get_landmarks(raw_predictions[self.landmarks_output[_idx]][batch_id],
                                                    anchor_num, anchors)[keep, :]
                    landmarks_list.extend(landmarks)
            scores = np.reshape(scores_list, -1)
            mask_scores = np.reshape(mask_scores_list, -1)
            labels = np.full_like(scores, 1, dtype=int)
            x_mins, y_mins, x_maxs, y_maxs = np.array(proposals_list).T # pylint: disable=E0633
            detection_representation = DetectionPrediction(
                identifier, labels, scores, x_mins / x_scale, y_mins / y_scale, x_maxs / x_scale, y_maxs / y_scale
            )
            if not self.landmarks_output and not self.type_scores_output:
                results.append(detection_representation)
                continue
            representations = {}
            representations['face_detection'] = detection_representation
            if self.type_scores_output:
                representations['mask_detection'] = AttributeDetectionPrediction(
                    identifier, labels, scores, mask_scores, x_mins / x_scale,
                    y_mins / y_scale, x_maxs / x_scale, y_maxs / y_scale
                )
            if self.landmarks_output:
                landmarks_x_coords = np.array(landmarks_list)[:, :, ::2].reshape(len(landmarks_list), -1) / x_scale
                landmarks_y_coords = np.array(landmarks_list)[:, :, 1::2].reshape(len(landmarks_list), -1) / y_scale
                representations['landmarks_regression'] = FacialLandmarksPrediction(identifier, landmarks_x_coords,
                                                                                    landmarks_y_coords)
            results.append(ContainerPrediction(representations))
        return results

    def _get_proposals(self, bbox_deltas, anchor_num, anchors):
        bbox_deltas = bbox_deltas.transpose((1, 2, 0))
        bbox_pred_len = bbox_deltas.shape[2] // anchor_num
        bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
        proposals = self.bbox_pred(anchors, bbox_deltas)
        return proposals

    @staticmethod
    def _get_scores(scores, anchor_num):
        scores = scores[anchor_num:, :, :]
        scores = scores.transpose((1, 2, 0)).reshape(-1)
        return scores

    @staticmethod
    def _get_mask_scores(type_scores, anchor_num):
        mask_scores = type_scores[anchor_num * 2:, :, :]
        mask_scores = mask_scores.transpose((1, 2, 0)).reshape(-1)
        return mask_scores

    def _get_landmarks(self, landmark_deltas, anchor_num, anchors):
        landmark_pred_len = landmark_deltas.shape[0] // anchor_num
        landmark_deltas = landmark_deltas.transpose((1, 2, 0)).reshape((-1, 5, landmark_pred_len // 5))
        landmark_deltas *= self.landmark_std
        landmarks = self.landmark_pred(anchors, landmark_deltas)
        return landmarks

    @staticmethod
    def bbox_pred(boxes, box_deltas):
        if boxes.shape[0] == 0:
            return np.zeros((0, box_deltas.shape[1]))

        boxes = boxes.astype(np.float, copy=False)
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
        ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
        dx = box_deltas[:, 0:1]
        dy = box_deltas[:, 1:2]
        dw = box_deltas[:, 2:3]
        dh = box_deltas[:, 3:4]
        pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
        pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
        pred_w = np.exp(dw) * widths[:, np.newaxis]
        pred_h = np.exp(dh) * heights[:, np.newaxis]
        pred_boxes = np.zeros(box_deltas.shape)
        pred_boxes[:, 0:1] = pred_ctr_x - 0.5 * (pred_w - 1.0)
        pred_boxes[:, 1:2] = pred_ctr_y - 0.5 * (pred_h - 1.0)
        pred_boxes[:, 2:3] = pred_ctr_x + 0.5 * (pred_w - 1.0)
        pred_boxes[:, 3:4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

        if box_deltas.shape[1] > 4:
            pred_boxes[:, 4:] = box_deltas[:, 4:]

        return pred_boxes

    @staticmethod
    def anchors_plane(height, width, stride, base_anchors):
        num_anchors = base_anchors.shape[0]
        all_anchors = np.zeros((height, width, num_anchors, 4))
        for iw in range(width):
            sw = iw * stride
            for ih in range(height):
                sh = ih * stride
                for k in range(num_anchors):
                    all_anchors[ih, iw, k, 0] = base_anchors[k, 0] + sw
                    all_anchors[ih, iw, k, 1] = base_anchors[k, 1] + sh
                    all_anchors[ih, iw, k, 2] = base_anchors[k, 2] + sw
                    all_anchors[ih, iw, k, 3] = base_anchors[k, 3] + sh

        return all_anchors

    @staticmethod
    def generate_anchors_fpn(cfg):
        def generate_anchors(base_size=16, ratios=(0.5, 1, 2), scales=2 ** np.arange(3, 6)):
            base_anchor = np.array([1, 1, base_size, base_size]) - 1
            ratio_anchors = _ratio_enum(base_anchor, ratios)
            anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales) for i in range(ratio_anchors.shape[0])])
            return anchors

        def _ratio_enum(anchor, ratios):
            w, h, x_ctr, y_ctr = _generate_wh_ctrs(anchor)
            size = w * h
            size_ratios = size / ratios
            ws = np.round(np.sqrt(size_ratios))
            hs = np.round(ws * ratios)
            anchors = _make_anchors(ws, hs, x_ctr, y_ctr)
            return anchors

        def _scale_enum(anchor, scales):
            w, h, x_ctr, y_ctr = _generate_wh_ctrs(anchor)
            ws = w * scales
            hs = h * scales
            anchors = _make_anchors(ws, hs, x_ctr, y_ctr)
            return anchors

        def _generate_wh_ctrs(anchor):
            w = anchor[2] - anchor[0] + 1
            h = anchor[3] - anchor[1] + 1
            x_ctr = anchor[0] + 0.5 * (w - 1)
            y_ctr = anchor[1] + 0.5 * (h - 1)
            return w, h, x_ctr, y_ctr

        def _make_anchors(ws, hs, x_ctr, y_ctr):
            ws = ws[:, np.newaxis]
            hs = hs[:, np.newaxis]
            anchors = np.hstack((
                x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1), x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
            ))
            return anchors

        rpn_feat_stride = [int(k) for k in cfg]
        rpn_feat_stride.sort(reverse=True)
        anchors = []
        for stride in rpn_feat_stride:
            feature_info = cfg[stride]
            bs = feature_info['BASE_SIZE']
            __ratios = np.array(feature_info['RATIOS'])
            __scales = np.array(feature_info['SCALES'])
            anchors.append(generate_anchors(bs, __ratios, __scales))

        return anchors

    @staticmethod
    def landmark_pred(boxes, landmark_deltas):
        if boxes.shape[0] == 0:
            return np.zeros((0, landmark_deltas.shape[1]))
        boxes = boxes.astype(np.float, copy=False)
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
        ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
        pred = landmark_deltas.copy()
        for i in range(5):
            pred[:, i, 0] = landmark_deltas[:, i, 0] * widths + ctr_x
            pred[:, i, 1] = landmark_deltas[:, i, 1] * heights + ctr_y

        return pred
