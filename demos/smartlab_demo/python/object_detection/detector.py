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
from .settings import MwGlobalExp
from thread_argument import ThreadWithReturnValue
from .subdetectors import SubDetector, CascadedSubDetector


class Detector:
    def __init__(self,
        core,
        device,
        top_models: list,
        side_models: list,
        backend: str='openvino'):

        '''Object Detection Variables'''
        self.object_detection_boxes = []
        self.front_anno_text_files = None
        self.top_anno_text_files = None

        '''configure output/preview'''
        self.save_result = False
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
        self.top_glb_exp = MwGlobalExp(
            core=core,
            device=device,
            num_classes = 10,
            model_path  = top_models[0],
            conf_thresh= 0.1,
            nms_thresh = 0.3)

        ###           topview.global_subdetector2          ###
        #  max-number constraints:
        #     "weights", 6; "tweezers", 1; "battery", 1;
        #  other conditions:
        #     conf 0.1; nms 0.2
        self.top_loc_exp = MwGlobalExp(
            core=core,
            device=device,
            num_classes = 4,
            model_path  = top_models[1],
            conf_thresh= 0.1,
            nms_thresh = 0.2,
            parent_obj = 'scale')

        '''configure settings for 2 models in front view'''
        ###          frontview.global_subdetector1           ###
        #  max-number constraints:
        #     "balance", 1; "weights",    omit; "tweezers", 1;
        #     "box"    , 1; "battery", 1; "tray"    , 2;
        #     "ruler"  , 1; "rider"  ,    omit; "scale"   , 1;
        #     "hand", 2;
        #  other conditions:
        #     conf 0.2; nms 0.3
        self.side_glb_exp = MwGlobalExp(
            core=core,
            device=device,
            num_classes = 10,
            model_path  = side_models[0],
            conf_thresh= 0.2,
            nms_thresh = 0.3)

        ###           frontview.global_subdetector2          ###
        #  max-number constraints:
        #     "weights", 6; "tweezers", 1; "battery", 1;
        #  other conditions:
        #     conf 0.1; nms 0.3
        self.side_loc_exp = MwGlobalExp(
            core=core,
            device=device,
            num_classes = 3,
            model_path  = side_models[1],
            conf_thresh= 0.1,
            nms_thresh = 0.3,
            parent_obj = 'ruler')

        ### concatenate list of class names for top/side views
        self.all_classes  = list(self.top_glb_exp.mw_classes)
        self.all_classes += list(self.top_loc_exp.mw_classes)
        self.all_classes += list(self.side_glb_exp.mw_classes)
        self.all_classes += list(self.side_loc_exp.mw_classes)
        self.all_classes = sorted(list(set(self.all_classes)))

        #  max-number constraints:
        self.max_nums = {
        "balance": 1, "weights": 6, "tweezers": 1,
        "box"    : 1, "battery": 1, "tray"    : 2,
        "ruler"  : 1, "rider"  : 1, "scale"   : 1,
        "hand"   : 2, "roundscrew1" : 2, "roundscrew2" : 2,
        "pointer": 1, "pointerhead" : 1}

        ### load models for top view
        self.top_glb_subdetector = SubDetector(self.top_glb_exp, self.all_classes)
        self.top_scale_subdetector = CascadedSubDetector(self.top_loc_exp, self.all_classes)
        ### load models for side view
        self.side_glb_subdetector = SubDetector(self.side_glb_exp, self.all_classes)
        self.side_ruler_subdetector = CascadedSubDetector(self.side_loc_exp, self.all_classes)

    def _get_parent_roi(self, preds, parent_id):
        for pred in preds:
            if parent_id == pred[-1]:
                res = pred[: 4]
                return res
        return None

    def _detect_one(self, img, view='top'):
        if view == 'top': # top view
            glb_subdet = self.top_glb_subdetector
            loc_subdet = self.top_scale_subdetector
        else: # side view
            glb_subdet = self.side_glb_subdetector
            loc_subdet = self.side_ruler_subdetector

        # global detector inference
        outputs = glb_subdet.inference(img)
        if len(outputs) == 0: return None, None, None
        all_preds = outputs[0][0]

        # # local detector inference
        # parent_cat = loc_subdet.parent_cat
        # parent_id = glb_subdet.detcls2id[parent_cat]
        # parent_roi = self._get_parent_roi(all_preds[-1], parent_id)
        # if parent_roi is not None: outputs = loc_subdet.inference_in(img, parent_roi)
        # else: outputs[0] = None
        # all_preds.append(outputs[0])

        # for i, sub_detector in enumerate([glb_subdet, loc_subdet]):
        #     if not hasattr(sub_detector, 'is_cascaded'):
        #         outputs = sub_detector.inference(img)
        #     else:
        #         if len(all_preds) == 0:
        #             continue

        #         parent_cat = sub_detector.parent_cat
        #         parent_id = glb_subdet.detcls2id[parent_cat]
        #         parent_roi = self._get_parent_roi(all_preds[-1], parent_id)

        #         if parent_roi is not None:
        #             outputs = sub_detector.inference_in(img, parent_roi)
        #         else:
        #             outputs[0] = None

        #     if outputs[0] is not None:
        #         preds = outputs[0] # work if bsize = 1
        #     else:
        #         continue
        #     all_preds.append(preds)

        for r, pred in enumerate(all_preds):
            cls_id = int(pred[-1])
            all_preds[r, -1] = cls_id

        # remap to original image scale
        bboxes = all_preds[:, :4]
        cls = all_preds[:, 6]
        scores = all_preds[:, 4] * all_preds[:, 5]

        return bboxes, cls, scores

    def inference(self, img_top, img_side):
        """
        Given input arrays for two view, need to generate and save 
            the corresponding detection results in the specific data structure.
        Args:
        img_top: img array of H x W x C for the top view
        img_front: img_array of H x W x C for the front view
        Returns:
        prediction results for the two images
        """

        ### sync mode ###
        top_bboxes, top_cls_ids, top_scores = self._detect_one(img_top, view='top')
        side_bboxes, side_cls_ids, side_scores = self._detect_one(img_side, view='side')

        # get class label
        top_labels = [self.all_classes[int(i)-1] for i in top_cls_ids]
        side_labels = [self.all_classes[int(i)-1] for i in side_cls_ids]

        return [top_bboxes, top_cls_ids, top_labels, top_scores], [side_bboxes, side_cls_ids, side_labels, side_scores]

    def inference_multithread(self, img_top, img_side):
        """
        Given input arrays for two view, need to generate and save the corresponding detection results
            in the specific data structure.
        Args:
        img_top: img array of H x W x C for the top view
        img_side: img_array of H x W x C for the side view
        Returns:
        prediction results for the two images
        """

        # creat detector thread and segmentor thread
        tdetTop = ThreadWithReturnValue(target = self._detect_one, args = (img_top, 'top',))
        tdetSide = ThreadWithReturnValue(target = self._detect_one, args = (img_side, 'side',))
        # start()
        tdetTop.start()
        tdetSide.start()
        # join()
        top_bboxes, top_cls_ids, top_scores = tdetTop.join()
        side_bboxes, side_cls_ids, side_scores = tdetSide.join()

        # get class label
        top_labels = [self.all_classes[int(i)-1] for i in top_cls_ids]
        side_labels = [self.all_classes[int(i)-1] for i in side_cls_ids]

        return [top_bboxes, top_cls_ids, top_labels, top_scores], [side_bboxes, side_cls_ids, side_labels, side_scores]
