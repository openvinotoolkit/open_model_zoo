# -*- coding: utf-8 -*-
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

import os
import cv2
import torch
from torch import nn

import sys
sys.path.append('object_detection')
from data.data_augment import ValTransform
from settings import MwGlobalExp
from tools.geometry import postprocess, bboxes_iou
from data.vis import vis


class SubDetector(object):
  # modified from source: https://github.com/Megvii-BaseDetection/YOLOX
  # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
  def __init__(
    self,
    model:nn.Module,
    exp:MwGlobalExp,
    cls_names:list,
    legacy=False,
  ):
    self.model = model
    self.cls_names = cls_names
    self.num_classes = exp.num_classes
    self.confthre = exp.confthre
    self.nmsthre = exp.nmsthre
    self.test_size = exp.test_size
    self.preproc = ValTransform(legacy=legacy)

  def inference(self, img):
    img_info = {"id": 0}
    if isinstance(img, str):
      img_info["file_name"] = os.path.basename(img)
      img = cv2.imread(img)
    else:
      img_info["file_name"] = None
    height, width = img.shape[:2]
    img_info["height"] = height
    img_info["width"] = width
    img_info["raw_img"] = img
    ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
    img_info["ratio"] = ratio
    img, _ = self.preproc(img, None, self.test_size)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.float()
    with torch.no_grad():
      outputs = self.model(img)
      outputs = postprocess(
        outputs, self.num_classes, self.confthre,
        self.nmsthre, class_agnostic=True
      )
    return outputs, img_info

  def visual(self, output, img_info, cls_conf=0.35):
    ratio = img_info["ratio"]
    img = img_info["raw_img"]
    if output is None:
      return img
    if isinstance(output, list):
      if output[0] is None:
        return img
      else:
        output = torch.cat(output)
    #output = output.to('cpu')
    bboxes = output[:, 0:4]
    # preprocessing: resize
    bboxes /= ratio
    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]
    vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
    return vis_res

  def pseudolabel(self, output, img_info, idx_offset, cls_conf=0.35):
    ratio = img_info["ratio"]
    img = img_info["raw_img"]
    image_id = img_info['id']
    if output is None:
      return img
    output = output.cpu()
    bboxes = output[:, 0:4]
    # preprocessing: resize
    bboxes /= ratio # [[x0,y0,x1,y1], ...]
    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]
    # enumerate all bbox
    i=0
    res = []
    for box, c, s in zip(bboxes, cls, scores):
      if s < cls_conf:
        continue
      x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
      w, h = x1-x0, y1-y0
      idx = idx_offset + i
      i += 1
      cat_id = int(c)
      res.append({
        'area': w * h, 
        'bbox': [x0, y0, w, h],
        'category_id': cat_id,
        'id': idx,
        'image_id': image_id,
        'iscrowd': 0,
        'segmentation': [[x0, y0, x1, y1]]
      })
    return res


def get_images(root_path):
  fimgs = []
  for f in os.listdir(root_path):
    if f.endswith('.PNG'):
      fimgs.append(os.path.join(root_path, f))
  return fimgs


class Detector(object):
  def __init__(self, fp_top_models:list, fp_front_models:list, is_show:bool):
    '''Object Detection Variables'''
    self.object_detection_boxes = []
    self.front_anno_text_files = None
    self.top_anno_text_files = None

    '''configure output/preview'''
    self.save_result = False
    self.is_show = is_show

    '''configure settings for 2 models in top view'''
    ###          topview.global_subdetector1           ###
    #  model: mw-a513v1_glb1cls10.pth
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
      root_input = None,
      conf_thresh= 0.1,
      nms_thresh = 0.3,
      is_show = self.is_show
    )
    ###           topview.global_subdetector2          ###
    #  model: mw-a513v1_glb2bcls3.pth
    #  max-number constraints:
    #     "weights", 6; "tweezers", 1; "battery", 1;      
    #  other conditions:
    #     conf 0.1; nms 0.2
    self.top2_exp = MwGlobalExp(
      num_classes = 3,
      fp_model = fp_top_models[1],
      root_input = None,
      conf_thresh= 0.1,
      nms_thresh = 0.2,
      is_show = self.is_show
    )
    
    '''configure settings for 2 models in front view'''
    ###          frontview.global_subdetector1           ###
    #  model: mw-a513v2_glb1cls10.pth
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
      root_input = None,
      conf_thresh= 0.2,
      nms_thresh = 0.3,
      is_show = self.is_show
    )
    ###           frontview.global_subdetector2          ###
    #  model: mw-a513v2_glb2bcls3.pth
    #  max-number constraints:
    #     "weights", 6; "tweezers", 1; "battery", 1;      
    #  other conditions:
    #     conf 0.1; nms 0.2
    self.front2_exp = MwGlobalExp(
      num_classes = 3,
      fp_model = fp_front_models[1],
      root_input = None,
      conf_thresh= 0.1,
      nms_thresh = 0.2,
      is_show = self.is_show
    )

    ### concatenate list of class names for topview
    cls1 = self.top1_exp.mw_classes
    cls2 = self.top2_exp.mw_classes
    self.classes = cls1 + cls2
    self.offset_cls_idx = [0, len(self.top1_exp.mw_classes)]
    #  max-number constraints:
    self.max_nums = {
      "balance":1, "weights":6, "tweezers":1,
      "box"    :1, "battery":1, "tray"    :2,
      "ruler"  :1, "rider"  :1, "scale"   :1,
      "hand":2,
    }
    ### build map of overlap classes
    self.cls2tocls1 = {}
    cls1_c2i = {c:i for i, c in enumerate(cls1)}
    cls2_c2i = {c:(i + self.offset_cls_idx[1]) for i, c in enumerate(cls2)}
    for cls in cls2:
      if cls in cls1:
        idx1 = cls1_c2i[cls]
        idx2 = cls2_c2i[cls]
        self.cls2tocls1[idx2] = idx1
    self.repeat_cls2_ids = self.cls2tocls1.keys()
    self.norepeat_num_classes = len(self.classes) - len(self.repeat_cls2_ids)

  def initialize(self):
    """
      todo Initialize the model & load the weights for two view
      todo Initialize the variables (specific data structure) for storing detection results.
    """
    # helper func
    def create_subdetector(exp:MwGlobalExp) -> SubDetector:
      det = exp.get_model()
      assert isinstance(det, nn.Module)
      det.eval()
      assert exp.fp_model.endswith('.pth')
      params_dict = torch.load(exp.fp_model, map_location='cpu')
      det.load_state_dict(params_dict['model'])
      return SubDetector(det, exp, exp.mw_classes, False)
    ### load models for top view
    self.top1_subdetector = create_subdetector(self.top1_exp)
    self.top2_subdetector = create_subdetector(self.top2_exp)
    ### load models for front view
    self.front1_subdetector = create_subdetector(self.front1_exp)
    self.front2_subdetector = create_subdetector(self.front2_exp)

  def _apply_detection_constraints(self, predictions:torch.Tensor, nmsthre=0.3):
    assert predictions.dim() == 2 and predictions.shape[1] == 7
    # sort by conf_score * cls_score
    sorted_preds = sorted(
      predictions, 
      key=lambda x:x[4]*x[5], 
      reverse=True)
    sorted_preds = torch.vstack(sorted_preds)
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
    clean_preds = torch.vstack(clean_preds)
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
      if isinstance(outputs[0], list):
        if outputs[0] is None:
          continue
        else:
          preds = torch.cat(outputs[0]) # work if bsize = 1
      else:
        preds = outputs[0]
      preds[:, 6] += self.offset_cls_idx[i]
      all_preds.append(preds)
    all_preds = torch.cat(all_preds)
    # merge same classes from model 2
    for r, pred in enumerate(all_preds):
      cls_id = int(pred[-1])
      if cls_id in self.repeat_cls2_ids:
        all_preds[r, -1] = self.cls2tocls1[cls_id]
    # post-process again on merged detections
    #shared_confthre = min(self.top1_exp.confthre, self.top2_exp.confthre)
    #shared_nmsthre = min(self.top1_exp.nmsthre, self.top2_exp.nmsthre)
    #all_preds = postprocess(all_preds.unsqueeze(dim=0), self.norepeat_num_classes, shared_confthre, shared_nmsthre)[0]
    # restrict object number for each class
    all_preds = self._apply_detection_constraints(all_preds)
    # remap to original image scale
    ratio = img_info['ratio']
    bboxes = all_preds[:, :4] / ratio
    cls = all_preds[:, 6]
    scores = all_preds[:, 4] * all_preds[:, 5]
    return bboxes, cls, scores

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

    top_bboxes = top_bboxes.detach().numpy()
    top_cls_ids = top_cls_ids.detach().numpy()
    top_scores = top_scores.detach().numpy()
    front_bboxes = front_bboxes.detach().numpy()
    front_cls_ids = front_cls_ids.detach().numpy()
    front_scores = front_scores.detach().numpy()
    # get class string
    top_cls_ids = [ self.classes[int(x)] for x in top_cls_ids ]
    front_cls_ids = [ self.classes[int(x)] for x in front_cls_ids ]

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