"""
 Copyright (C) 2021-2022 Intel Corporation

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
from .preprocess import preprocess
from .settings import MwGlobalExp
from .deploy_util import multiclass_nms, demo_postprocess
from .subdetectors import SubDetector
from .vis import vis


class Detector(object):
    def __init__(self,
        ie,
        device,
        fp_top_models: list,
        fp_front_models: list,
        is_show: bool,
        backend: str='openvino'):

        '''Object Detection Variables'''
        self.object_detection_boxes = []
        self.front_anno_text_files = None
        self.top_anno_text_files = None

        '''configure output/preview'''
        self.save_result = False
        self.is_show = is_show
        self.backend = backend

        '''configure settings for 2 models in top view'''
        ###          topview.global_subdetector1           ###
        #  max-number constraints:
        #     "balance", 1; "weights",    omit; "tweezers", 1;
        #     "box"    , 1; "battery", 1; "tray"    , 2;
        #     "ruler"  , 1; "rider"  ,    omit; "scale"   , 1;
        #     "hand", 2;
        #  other conditions:
        #     conf 0.1; nms 0.3
        self.top1_exp = MwGlobalExp(
            num_classes = 10,
            fp_model = fp_top_models[0],
            conf_thresh= 0.1,
            nms_thresh = 0.3,
            ie=ie,
            device=device)

        ###           topview.global_subdetector2          ###
        #  max-number constraints:
        #     "weights", 6; "tweezers", 1; "battery", 1;
        #  other conditions:
        #     conf 0.1; nms 0.2
        self.top2_exp = MwGlobalExp(
            num_classes = 3,
            fp_model = fp_top_models[1],
            conf_thresh= 0.1,
            nms_thresh = 0.2,
            ie=ie,
            device=device)

        '''configure settings for 2 models in front view'''
        ###          frontview.global_subdetector1           ###
        #  max-number constraints:
        #     "balance", 1; "weights",    omit; "tweezers", 1;
        #     "box"    , 1; "battery", 1; "tray"    , 2;
        #     "ruler"  , 1; "rider"  ,    omit; "scale"   , 1;
        #     "hand", 2;
        #  other conditions:
        #     conf 0.2; nms 0.3
        self.front1_exp = MwGlobalExp(
            num_classes = 10,
            fp_model = fp_front_models[0],
            conf_thresh= 0.2,
            nms_thresh = 0.3,
            ie=ie,
            device=device)

        ###           frontview.global_subdetector2          ###
        #  max-number constraints:
        #     "weights", 6; "tweezers", 1; "battery", 1;
        #  other conditions:
        #     conf 0.1; nms 0.2
        self.front2_exp = MwGlobalExp(
            num_classes = 3,
            fp_model = fp_front_models[1],
            conf_thresh= 0.1,
            nms_thresh = 0.2,
            ie=ie,
            device=device)

        ### concatenate list of class names for topview
        cls1 = self.top1_exp.mw_classes
        cls2 = self.top2_exp.mw_classes
        self.classes = cls1 + cls2
        self.offset_cls_idx = [0, len(self.top1_exp.mw_classes)]
        #  max-number constraints:
        self.max_nums = {
            "balance": 1, "weights": 6, "tweezers": 1,
            "box"    : 1, "battery": 1, "tray"    : 2,
            "ruler"  : 1, "rider"  : 1, "scale"   : 1,
            "hand": 2,
        }

        ### build map of overlap classes
        self.cls2tocls1 = {}
        cls1_c2i = {c: i for i, c in enumerate(cls1)}
        cls2_c2i = {c: (i + self.offset_cls_idx[1]) for i, c in enumerate(cls2)}
        for cls in cls2:
            if cls in cls1:
                idx1 = cls1_c2i[cls]
                idx2 = cls2_c2i[cls]
                self.cls2tocls1[idx2] = idx1
        self.repeat_cls2_ids = self.cls2tocls1.keys()
        self.norepeat_num_classes = len(self.classes) - len(self.repeat_cls2_ids)

        ### load models for top view
        self.top1_subdetector = SubDetector(self.top1_exp, self.backend)
        self.top2_subdetector = SubDetector(self.top2_exp, self.backend)
        ### load models for front view
        self.front1_subdetector = SubDetector(self.front1_exp, self.backend)
        self.front2_subdetector = SubDetector(self.front2_exp, self.backend)

    def _apply_detection_constraints(self, predictions: np.ndarray, nmsthre=0.3):
        ### sort by conf_score * cls_score
        sorted_preds = sorted(
            predictions,
            key=lambda x: x[4]*x[5],
            reverse=True)
        sorted_preds = np.vstack(sorted_preds)
        ### reserve indicated number foreach cls
        classes = list(self.max_nums.keys())
        ulimits = list(self.max_nums.values())
        res = [[] for _ in classes]
        for pred in sorted_preds:
            cls = int(pred[-1]) # index by class
            n_res = len(res[cls])
            if n_res < ulimits[cls]:
                res[cls].append(pred) # max reserve <= ulimit
        clean_preds = []
        [clean_preds.extend(v) for v in res]
        clean_preds = np.vstack(clean_preds)

        return clean_preds

    def _detect_one(self, img, view='top'):
        if view == 'top': # top view
            sub_detector1 = self.top1_subdetector
            sub_detector2 = self.top2_subdetector
        else: # front view
            sub_detector1 = self.front1_subdetector
            sub_detector2 = self.front2_subdetector
        all_preds = []
        for i, sub_detector in enumerate([sub_detector1, sub_detector2]):
            outputs, img_info = sub_detector.inference(img)
            if outputs[0] is not None:
                preds = outputs[0] # work if bsize = 1
            else:
                continue
            preds[:, 6] += self.offset_cls_idx[i]
            all_preds.append(preds)

        if len(all_preds) > 0:
            all_preds = np.concatenate(all_preds)
        else:# in case of no obj detected
            all_preds = np.zeros((1, 7))

        # merge same classes from model 2
        for r, pred in enumerate(all_preds):
            cls_id = int(pred[-1])
            if cls_id in self.repeat_cls2_ids:
                all_preds[r, -1] = self.cls2tocls1[cls_id]

        # restrict object number for each class
        all_preds = self._apply_detection_constraints(all_preds)

        # remap to original image scale
        ratio = img_info['ratio']
        bboxes = all_preds[:, :4] / ratio
        cls = all_preds[:, 6]
        scores = all_preds[:, 4] * all_preds[:, 5]

        return bboxes, cls, scores

    def pseudo_detect(self, origin_img, input_blob, out_blob, exec_net, h, w, idx_offset: int):
        image, ratio = preprocess(origin_img, (h, w))
        res = exec_net.infer(inputs={input_blob: image})
        res = res[out_blob]

        predictions = demo_postprocess(res, (h, w), p6=False)[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4, None] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2. # x0,y0,x1,y1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)

        if dets is not None:
            final_boxes = dets[:, :4]
            final_scores, final_cls_inds = dets[:, 4], dets[:, 5]
        else:
            return idx_offset, []

        return final_boxes, final_cls_inds, final_scores


    def inference(self, img_top, img_front):
        """
        todo Given input arrays for two view, need to generate and save the corresponding detection results
            in the specific data structure.
        Args:
        img_top: img array of H x W x C for the top view
        img_front: img_array of H x W x C for the front view

        Returns:
        prediction results for the two images
        """
        top_bboxes, top_cls_ids, top_scores = self._detect_one(img_top, view='top')
        front_bboxes, front_cls_ids, front_scores = self._detect_one(img_front, view='front')

        # get class string
        top_cls_ids = [ self.classes[int(x)] for x in top_cls_ids ]
        front_cls_ids = [ self.classes[int(x)] for x in front_cls_ids ]


        # return [], []
        if self.is_show:
            vis_top = vis(
                img_top,
                top_bboxes,
                top_scores,
                top_cls_ids,
                self.top1_exp.confthre,
                self.classes)
            vis_front = vis(
                img_front,
                front_bboxes,
                front_scores,
                front_cls_ids,
                self.front1_exp.confthre,
                self.classes)
            return vis_top, vis_front
        else:
            return [top_bboxes, top_cls_ids, top_scores], [front_bboxes, front_cls_ids, front_scores]
