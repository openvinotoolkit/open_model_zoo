
# -*- coding:utf-8 -*-
import time
import torch
import numpy as np
from torch import Tensor
from collections import defaultdict


def NMS(
  boxs:Tensor,
  scores:Tensor, 
  iou_thd:float = 0.01,
)-> Tensor:
  #Qi Sun, @intel SMG
  """
  Param:
    boxs:[N,4] x1,y1,x2,y2
    scores:[N]
    iou_thd: ious threhold
  Return:
    Tensor(boxs)

  """
  x1 = boxs[:,0]
  y1 = boxs[:,1]
  x2 = boxs[:,2]
  y2 = boxs[:,3]
  areas = (x2 - x1 + 1) * (y2 - y1 + 1)
  order = scores.argsort()
  detection = []
  while order.numel()>0:
    i = order[-1]
    detection.append(i)
    xx1 = np.maximum(x1[i], x1[order[:-1]])
    yy1 = np.maximum(y1[i], y1[order[:-1]])
    xx2 = np.minimum(x2[i], x2[order[:-1]])
    yy2 = np.minimum(y2[i], y2[order[:-1]])
    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    ious = inter / (areas[i] + areas[order[1:]] - inter)
    idx = np.where(ious <= iou_thd)[0]
    order = order[idx]
  detection = torch.tensor(detection)
  return boxs[detection]  


def soft_NMS(
  boxs:Tensor, 
  scores:Tensor, 
  iou_thd:float = 0.7,
  score_thd:float = 0.05,
  sigma:float = 0.5,
  method:str = 'linear'
)-> Tensor:
  #Qi Sun, @intel SMG
  """
  Param:
    boxs:[N,4] x1,y1,x2,y2
    scores:[N]
    iou_thd: ious threhold
    score_thd: confidence score
    sigma: gaussian arg
    method: linear or gaussian
  Return:
    Tensor(boxs)
  """
  x1 = boxs[:,0]
  y1 = boxs[:,1]
  x2 = boxs[:,2]
  y2 = boxs[:,3]
  areas = (x2 - x1 + 1) * (y2 - y1 + 1)
  order = scores.argsort()
  detection = []
  while order.numel()>0:
    i = order[-1]
    order = order[:-1]
    xx1 = np.maximum(x1[i], x1[order])
    yy1 = np.maximum(y1[i], y1[order])
    xx2 = np.minimum(x2[i], x2[order])
    yy2 = np.minimum(y2[i], y2[order])
    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    ious = inter / (areas[i] + areas[order] - inter)
    idx = np.where(ious <= iou_thd)[0]
    if method == 'linear':
      scores[idx] = scores[idx] * (1 - ious[idx])
    else:
      scores[idx] = scores[idx] * torch.exp(-torch.square(ious[idx])/sigma)
  detection = torch.tensor(np.where(scores >= score_thd)[0])
  return boxs[detection]  


def confluence(
  bounding_boxes,
  scores,
  confluence_thr=0.7,
  gaussian=False,
  score_thr=0.05,
  sigma=0.5
  )->Tensor:
  # original source:https://github.com/ashep29/confluence
  # copy right: No licence
  """
  Parameters:
    bounding_boxes: list of bounding boxes (x1,y1,x2,y2)
    scores: list of class confidence scores (0.0-1.0)
    confluence_thr: value between 0 and 2, with optimum from 0.5-0.8
    gaussian: boolean switch to turn gaussian decaying of suboptimal bounding box confidence scores (setting to False results in suppression of suboptimal boxes)
    score_thr: class confidence score
    sigma: used in gaussian decaying. A smaller value causes harsher decaying.
  Returns:
    Tensor(boxs)
  """
  def normalise_coordinates(x1, y1, x2, y2,min_x,max_x,min_y,max_y): ### normalise 
    """
    Parameters: 
      x1, y1, x2, y2: bounding box coordinates to normalise
      min_x,max_x,min_y,max_y: minimum and maximum bounding box values (min = 0, max = 1)
    Returns:
      Normalised bounding box coordinates (scaled between 0 and 1)
    """
    x1, y1, x2, y2 = (x1-min_x)/(max_x-min_x), (y1-min_y)/(max_y-min_y), (x2-min_x)/(max_x-min_x), (y2-min_y)/(max_y-min_y)
    return x1, y1, x2, y2

  def assign_boxes_to_classes(bounding_boxes, scores):
    """
    Parameters: 
        bounding_boxes: list of bounding boxes (x1,y1,x2,y2)
        classes: list of class identifiers (int value, e.g. 1 = person)
        scores: list of class confidence scores (0.0-1.0)
    Returns:
        boxes_to_classes: defaultdict(list) containing mapping to bounding boxes and confidence scores to class
    """
    boxes_to_classes = defaultdict(list)
    for each_box, each_score in zip(bounding_boxes, scores):
      if each_score >= 0.05:
        boxes_to_classes[0].append(np.array([each_box[0],each_box[1],each_box[2],each_box[3], each_score]))
    return boxes_to_classes

  class_mapping = assign_boxes_to_classes(bounding_boxes, scores)
  ### gather bounding_boxs, classes,scores into list->class_mapping
  output = {}
  # for each_class in class_mapping: ### for c_i in C
  dets = np.array(class_mapping[0])
  retain = [] ### initialize detection boxes & orignial boxes
  while dets.size > 0: ### while B != empty
    confluence_scores,proximities = [],[] ### initialize b_s, s_s
    while len(confluence_scores)<np.size(dets,0): ### for b_i, s_i in B,S:
      ### compute P between b_i and others
      current_box = len(confluence_scores)
      x1, y1, x2, y2 = dets[current_box, 0], dets[current_box, 1], dets[current_box, 2], dets[current_box, 3]
      confidence_score = dets[current_box, 4]
      xx1,yy1,xx2,yy2,cconf = dets[np.arange(len(dets))!=current_box, 0],dets[np.arange(len(dets))!=current_box, 1],dets[np.arange(len(dets))!=current_box, 2],dets[np.arange(len(dets))!=current_box, 3],dets[np.arange(len(dets))!=current_box, 4]
      min_x,min_y,max_x,max_y = np.minimum(x1, xx1),np.minimum(y1, yy1),np.maximum(x2, xx2),np.maximum(y2, yy2)    
      x1, y1, x2, y2 = normalise_coordinates(x1, y1, x2, y2,min_x,max_x,min_y,max_y)
      xx1, yy1, xx2, yy2 = normalise_coordinates(xx1, yy1, xx2, yy2,min_x,max_x,min_y,max_y)
      hd_x1,hd_x2,vd_y1,vd_y2 = abs(x1-xx1),abs(x2-xx2),abs(y1-yy1),abs(y2-yy2)
      proximity = (hd_x1+hd_x2+vd_y1+vd_y2)
      all_proximities = np.ones_like(proximity)
      cconf_scores = np.zeros_like(cconf)
      ### if P<2 then update confluence
      all_proximities[proximity <= confluence_thr] = proximity[proximity <= confluence_thr]
      cconf_scores[proximity <= confluence_thr]=cconf[proximity <= confluence_thr]
      if(cconf_scores.size>0):
          confluence_score = np.amax(cconf_scores)
      else:
          confluence_score = confidence_score
      if(all_proximities.size>0):
          proximity = (sum(all_proximities)/all_proximities.size)*(1-confidence_score)
      else:
          proximity = sum(all_proximities)*(1-confidence_score)
      confluence_scores.append(confluence_score)
      proximities.append(proximity)
    ### end for
    conf = np.array(confluence_scores)
    prox = np.array(proximities)
    ### B_f,S_f,C_f || b_s...c_i
    dets_temp = np.concatenate((dets, prox[:, None]), axis=1)
    dets_temp = np.concatenate((dets_temp, conf[:, None]), axis=1)
    min_idx = np.argmin(dets_temp[:, 5], axis=0) ### find s_i
    dets[[0, min_idx], :] = dets[[min_idx, 0], :] ### find s_i
    dets_temp[[0, min_idx], :] = dets_temp[[min_idx, 0], :]
    dets[0,4]=dets_temp[0,6]
    retain.append(dets[0, :])
    ### P <- proximity(b,b_s)
    x1, y1, x2, y2 = dets[0, 0], dets[0, 1], dets[0, 2], dets[0, 3]
    min_x = np.minimum(x1, dets[1:, 0])
    min_y = np.minimum(y1, dets[1:, 1])
    max_x = np.maximum(x2, dets[1:, 2])   
    max_y = np.maximum(y2, dets[1:, 3])
    x1, y1, x2, y2 = normalise_coordinates(x1, y1, x2, y2,min_x,max_x,min_y,max_y)
    xx1, yy1, xx2, yy2 = normalise_coordinates(dets[1:, 0], dets[1:, 1], dets[1:, 2], dets[1:, 3],min_x,max_x,min_y,max_y)
    md_x1,md_x2,md_y1,md_y2 = abs(x1-xx1),abs(x2-xx2),abs(y1-yy1),abs(y2-yy2) 
    manhattan_distance = (md_x1+md_x2+md_y1+md_y2)
    weights = np.ones_like(manhattan_distance)
    ### if P<M_d then B,S <- B-b, S-s
    if (gaussian == True):
        gaussian_weights = np.exp(-((1-manhattan_distance) * (1-manhattan_distance)) / sigma)
        weights[manhattan_distance<=confluence_thr]=gaussian_weights[manhattan_distance<=confluence_thr]
    else:
        weights[manhattan_distance<=confluence_thr]=manhattan_distance[manhattan_distance<=confluence_thr]
    dets[1:, 4] *= weights
    to_reprocess = np.where(dets[1:, 4] >= score_thr)[0]
    dets = dets[to_reprocess + 1, :]    
  output = torch.tensor(retain)
  output = output[:,:-1]
  return output


# def test():
#     # boxes and boxscores
#     boxes = torch.tensor([[200, 200, 400, 400],
#                           [220, 220, 420, 420],
#                           [200, 240, 400, 440],
#                           [240, 200, 440, 400],
#                           [1, 1, 2, 2]], dtype=torch.float)
#     boxscores = torch.tensor([0.8, 0.7, 0.6, 0.5, 0.1], dtype=torch.float)

#     a = soft_NMS(boxes,boxscores)
#     print("soft_NMS:\n",a)
#     b = NMS(boxes,boxscores)
#     print("NMS:\n",b)
#     c = confluence(boxes,boxscores)
#     print("confluence:\n",a)
# if __name__ == '__main__':
#     test()
