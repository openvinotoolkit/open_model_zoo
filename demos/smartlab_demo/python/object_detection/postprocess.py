import numpy as np
import torch
import torchvision

def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
  # original source: https://github.com/Megvii-BaseDetection/YOLOX
  # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
  prediction = torch.tensor(prediction)
  box_corner = prediction.new(prediction.shape)
  box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
  box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
  box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
  box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
  prediction[:, :, :4] = box_corner[:, :, :4]
  output = [None for _ in range(len(prediction))]
  for i, image_pred in enumerate(prediction):
    # If none are remaining => process next image
    if not image_pred.size(0):
      continue
    # Get score and class with highest confidence
    class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
    conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
    # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
    detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
    detections = detections[conf_mask]
    if not detections.size(0):
      continue
    if class_agnostic:
      nms_out_index = torchvision.ops.nms(
        detections[:, :4],
        detections[:, 4] * detections[:, 5],
        nms_thre,
      )
    else:
      nms_out_index = torchvision.ops.batched_nms(
        detections[:, :4],
        detections[:, 4] * detections[:, 5],
        detections[:, 6],
        nms_thre,
      )
    detections = detections[nms_out_index]
    if output[i] is None:
      output[i] = detections
    else:
      output[i] = torch.cat((output[i], detections))
  return output

def nms(bounding_boxes, confidence_score, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)
    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]
    # Confidence scores of bounding boxes
    score = np.array(confidence_score)
    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)
    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    nms_out_index = []
    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]
        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])
        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h
        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)
        left = np.where(ratio < threshold)
        order = order[left]

        nums_out = np.where(ratio >= threshold)
        nms_out_index.append(nums_out)

    return np.array(nms_out_index)