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
import math
import numpy as np

from ..adapters import Adapter
from ..config import NumberField
from ..representation import DetectionPrediction

class PersonVehicleDetectionAdapter(Adapter):
    __provider__ = 'person_vehicle_detection'
    predcition_types = (DetectionPrediction, )
    class_threshold_ = [0.0, 0.60, 0.85, 0.90, 0.70, 0.75, 0.80, 0.75]
    kMinObjectSize = 25

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'iou_threshold': NumberField(
                value_type=float, min_value=0, max_value=1, default=0.45, optional=True,
                description='Iou threshold for NMS')
        })
        return parameters

    def configure(self):
        self.iou_threshold = self.get_value_from_config('iou_threshold')

    def process(self, raw, identifiers, frame_meta):
        result = []
        if isinstance(raw, dict):
            bbox_pred = raw['bbox_pred']
            proposals = raw['proposals']
            cls_score = raw['cls_score']
            props_map = self.output_to_proposals(bbox_pred, proposals, cls_score, frame_meta[0])
            pred_items = self.get_proposals(props_map)
            result.append(pred_items)
        else:
            for batch_index in range(len(identifiers)):
                bbox_pred = raw[batch_index]['bbox_pred']
                proposals = raw[batch_index]['proposals']
                cls_score = raw[batch_index]['cls_score']
                props_map = self.output_to_proposals(bbox_pred, proposals, cls_score, frame_meta[batch_index])
                pred_items = self.get_proposals(props_map)
                result.append(pred_items)
        return result

    def output_to_proposals(self, bbox_pred, proposals, cls_score, frame_meta):
        img_height, img_width, _ = frame_meta['image_size']
        input_height, input_width, _ = frame_meta['image_info']

        ww = img_width
        hh = img_height
        sx = 1.0 / (input_width / img_width)
        sy = 1.0 / (input_height / img_height)
        num_rois = 96
        num_classes = 8

        # merge scores - Car: 3, Truck: 7, Van: 6
        for r in range(num_rois):
            indices = [3, 6, 7]

            def get_cls_score(idx, x=r):
                return cls_score[x][idx]

            indices.sort(key=get_cls_score, reverse=True)

            sum_score = 0.0
            for it in indices:
                sum_score += cls_score[r][it]
                cls_score[r][it] = 0.0

            cls_score[r][indices[0]] = sum_score

        # proposals map [class id, rectangle]
        props_map = []

        # NMS in tensorpack way
        for c in range(1, num_classes):
            scores = []
            boxes = []

            for r in range(num_rois):
                # class score
                scores.append(cls_score[r][c])
                # regressed bbox
                reg = bbox_pred[r][4*c:]
                roi = proposals[r][1:]
                boxes.append(self.regress_scale_clip_bbox(ww, hh, sx, sy, roi, reg))

            # nms in a class
            kidx = self.nms_apply(c, scores, boxes)

            # final class and box
            p_map = []
            for i in kidx:
                p_map.append([c, scores[i], boxes[i]])  # class_id, score, box_rectangle

            props_map.append(p_map)

        return props_map

    def get_proposals(self, props_map):
        vehicles = []
        detections = {'labels': [], 'scores': [], 'x_mins': [], 'y_mins': [], 'x_maxs': [], 'y_maxs': []}

        for props in props_map:
            if len(props) == 0:
                continue

            for prop in props:
                class_id = prop[0]
                rect = prop[2]
                score = prop[1]
                if (rect[2] - rect[0]) < self.kMinObjectSize or (rect[3] - rect[1]) < self.kMinObjectSize:
                    continue

                gt_class_id = self.convert_classid_to_gt_type(class_id)

                # person
                if gt_class_id == 0:
                    detections['labels'].append(gt_class_id)
                    detections['scores'].append(score)
                    detections['x_mins'].append(rect[0])
                    detections['y_mins'].append(rect[1])
                    detections['x_maxs'].append(rect[2])
                    detections['y_maxs'].append(rect[3])

                # vehicle
                else:
                    vehicles.append(prop)

        # sort by score and NMS in 2 wheels, 4 heels
        vehicles = self.nms_apply_prop(vehicles)

        # Convert to detection representation
        for vehicle in vehicles:
            rect = vehicle[2]
            score = vehicle[1]
            detections['labels'].append(1)
            detections['scores'].append(score)
            detections['x_mins'].append(rect[0])
            detections['y_mins'].append(rect[1])
            detections['x_maxs'].append(rect[2])
            detections['y_maxs'].append(rect[3])

        return DetectionPrediction(
            labels=detections['labels'],
            scores=detections['scores'],
            x_mins=detections['x_mins'],
            y_mins=detections['y_mins'],
            x_maxs=detections['x_maxs'],
            y_maxs=detections['y_maxs']
        )

    @staticmethod
    def regress_scale_clip_bbox(ww, hh, sx, sy, roi, reg):
        cx = (roi[0] + roi[2]) * 0.5
        cy = (roi[1] + roi[3]) * 0.5
        aw = (roi[2] - roi[0])
        ah = (roi[3] - roi[1])

        cx += (reg[0] * 0.1 * aw)
        cy += (reg[1] * 0.1 * ah)
        aw *= (math.exp(reg[2] * 0.2) * 0.5)
        ah *= (math.exp(reg[3] * 0.2) * 0.5)

        res = [min(ww, max(0.0, (cx - aw) * sx)),
               min(hh, max(0.0, (cy - ah) * sy)),
               min(ww, max(0.0, (cx + aw) * sx)),
               min(hh, max(0.0, (cy + ah) * sy))]

        return res

    def nms_apply(self, class_id, scores, boxes):
        filtered = self.filter_by_score(self.class_threshold_[class_id], scores)
        idx = filtered[0]
        is_dead = filtered[1]

        # iou based filter(nms)
        keep_ids = []

        for i, _ in enumerate(idx):
            li = idx[i]
            if is_dead[li]:
                continue

            keep_ids.append(li)

            for j in range(i + 1, len(idx)):
                ri = idx[j]

                if is_dead[ri]:
                    continue

                iou = self.compute_iou(boxes[li], boxes[ri])
                if iou > self.iou_threshold:
                    is_dead[ri] = 1

        return keep_ids

    def nms_apply_prop(self, input_props):
        idx = []
        keep_ids = []
        is_dead = []

        for i in range(len(input_props)):
            idx.append(i)
            is_dead.append(False)

        def get_score(ii):
            val = input_props[ii][1]    # body_bbox_score
            return val

        idx.sort(key=get_score, reverse=True)

        for i in range(len(input_props)):
            li = idx[i]
            if is_dead[li]:
                continue

            keep_ids.append(li)

            for j in range(i+1, len(input_props)):
                ri = idx[j]

                if is_dead[ri]:
                    continue

                iou = self.compute_iou_prop(input_props[li], input_props[ri])

                if iou > self.iou_threshold:
                    is_dead[ri] = True

        output_props = []
        for it in keep_ids:
            if is_dead[it]:
                continue
            output_props.append(input_props[it])

        return output_props

    @staticmethod
    def filter_by_score(threshold, scores):
        # Argsort by score
        idx = []
        is_dead = []
        for i in range(len(scores)):
            idx.append(i)
            is_dead.append(True)

        def get_score(ii):
            val = scores[ii]
            return val

        idx.sort(key=get_score, reverse=True)

        # Score based filter
        r_idx = []
        for i in idx:
            if scores[i] < threshold:
                is_dead.append(True)
                continue
            r_idx.append(i)
            is_dead[i] = False

        return [r_idx, is_dead]

    @staticmethod
    def compute_iou(lhs, rhs):
        ix = min(lhs[2], rhs[2]) - max(lhs[0], rhs[0])
        iy = min(lhs[3], rhs[3]) - max(lhs[1], rhs[1])
        ai = max(0.0, ix) * max(0.0, iy)

        al = (lhs[2] - lhs[0]) * (lhs[3] - lhs[1])
        ar = (rhs[2] - rhs[0]) * (rhs[3] - rhs[1])
        uu = al + ar - ai   # Union

        return ai / max(uu, 0.0000001)

    @staticmethod
    def compute_iou_prop(lhs, rhs):
        lhs_box = lhs[2]
        rhs_box = rhs[2]
        return PersonVehicleDetectionAdapter.compute_iou(lhs_box, rhs_box)

    @staticmethod
    def convert_classid_to_gt_type(class_id):
        return 0 if class_id == 5 else 1

class PersonVehicleDetectionRefinementAdapter(Adapter):
    __provider__ = 'person_vehilce_detection_refinement'
    predcition_types = (DetectionPrediction, )

    def process(self, raw, identifiers=None, frame_meta=None):
        thresholds = {1: 0.5, 2: 0.4, 3: 0.4, 4: 0.3, 5: 0.6, 6: 0.5, 7: 0.3}
        kBoxNormalize = [10.0, 10.0, 5.0, 5.0]

        detections = {'labels': [], 'scores': [], 'x_mins': [], 'y_mins': [], 'x_maxs': [], 'y_maxs': []}
        proposals = frame_meta[0]['candidates']
        img_height, img_width, _ = frame_meta[0]['image_size']
        identifier = identifiers[0]

        if proposals.x_mins.size == 0:
            return [DetectionPrediction(identifier=identifier)]

        for batch_index, prediction in enumerate(raw):
            cls_prob = prediction['final_prob'][0]
            bbox_pred = prediction['final_boxes'][0]

            max_cls_index = np.argmax(cls_prob)
            max_cls_prob = cls_prob[max_cls_index]

            if max_cls_index != 0:
                bbox_pred_index = max_cls_index
                x_min = proposals.x_mins[batch_index]
                y_min = proposals.y_mins[batch_index]
                x_max = proposals.x_maxs[batch_index]
                y_max = proposals.y_maxs[batch_index]

                x_min, y_min, x_max, y_max = self.transform_bbox_inv(
                    bbox_pred[bbox_pred_index][0] / kBoxNormalize[0], bbox_pred[bbox_pred_index][1] / kBoxNormalize[1],
                    bbox_pred[bbox_pred_index][2] / kBoxNormalize[2], bbox_pred[bbox_pred_index][3] / kBoxNormalize[3],
                    x_min, y_min, x_max, y_max,
                    img_width, img_height
                )
                self.update_by_roi(proposals, batch_index, x_min, y_min, x_max, y_max)
                if max_cls_prob > thresholds[proposals.labels[batch_index]]:
                    detections['scores'].append(max_cls_prob)
                    detections['labels'].append(max_cls_index)
                    detections['x_mins'].append(proposals.x_mins[batch_index])
                    detections['y_mins'].append(proposals.y_mins[batch_index])
                    detections['x_maxs'].append(proposals.x_maxs[batch_index])
                    detections['y_maxs'].append(proposals.y_maxs[batch_index])

        detections['labels'] = list(map(self.convert_classid_to_gt_type, detections['labels']))
        return [
            DetectionPrediction(
                identifier=identifier,
                labels=detections['labels'],
                scores=detections['scores'],
                x_mins=detections['x_mins'],
                y_mins=detections['y_mins'],
                x_maxs=detections['x_maxs'],
                y_maxs=detections['y_maxs']
            )
        ]

    @staticmethod
    def update_by_roi(proposals, batch_index, x_min, y_min, x_max, y_max):

        cx1 = (x_min + x_max) * 0.5
        cy1 = (y_min + y_max) * 0.5
        w1 = (x_max - x_min)
        h1 = (y_max - y_min)

        cx0 = (proposals.x_maxs[batch_index] + proposals.x_mins[batch_index]) * 0.5
        cy0 = (proposals.y_maxs[batch_index] + proposals.y_mins[batch_index]) * 0.5
        w0 = proposals.x_maxs[batch_index] - proposals.x_mins[batch_index]
        h0 = proposals.y_maxs[batch_index] - proposals.y_mins[batch_index]

        wx = w1/w0
        wy = h1/h0

        proposals.x_mins[batch_index] = int((proposals.x_mins[batch_index] - cx0) * wx + cx1)
        proposals.y_mins[batch_index] = int((proposals.y_mins[batch_index] - cy0) * wy + cy1)
        proposals.x_maxs[batch_index] = int((proposals.x_maxs[batch_index] - cx0) * wx + cx1)
        proposals.y_maxs[batch_index] = int((proposals.y_maxs[batch_index] - cy0) * wy + cy1)

    @staticmethod
    def transform_bbox_inv(dx, dy, d_log_w, d_log_h, x1, y1, x2, y2, img_w, img_h):
        # width & height of box
        w = x2 - x1 + 1.0
        h = y2 - y1 + 1.0
        ctr_x = x1 + 0.5 * w
        ctr_y = y1 + 0.5 * h

        # new center location according to gradient (dx, dy)
        pred_ctr_x = dx * w + ctr_x
        pred_ctr_y = dy * h + ctr_y

        # new width & height according to gradient d(log w), d(log h)
        pred_w = math.exp(d_log_w) * w
        pred_h = math.exp(d_log_h) * h

        # update upper-left corner location
        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h

        # update lower-right corner location
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h

        # adjust new corner locations to be within the image region,
        x1 = max(0, min(x1, img_w - 1.0))
        y1 = max(0, min(y1, img_h - 1.0))
        x2 = max(0, min(x2, img_w - 1.0))
        y2 = max(0, min(y2, img_h - 1.0))

        return x1, y1, x2, y2

    @staticmethod
    def convert_classid_to_gt_type(class_id):
        return 0 if class_id == 5 else 1
