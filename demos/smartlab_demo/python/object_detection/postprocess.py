import numpy as np
from deploy_util import nms, multiclass_nms


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    # original source: https://github.com/Megvii-BaseDetection/YOLOX
    # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
    box_corner = np.zeros(prediction.shape)

    # box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]
    output = [None for _ in range(len(prediction))]

    for i, image_pred in enumerate(prediction):
        # If none are remaining => process next image
        if len(image_pred[0]) == 0:
            continue
        # Get score and class with highest confidence
        class_conf = np.max(image_pred[:, 5:5+num_classes], axis=1, keepdims=True)
        class_pred = np.argmax(image_pred[: , 5:5+num_classes], axis=1)
        class_pred = np.expand_dims(class_pred, axis=1)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = np.concatenate((image_pred[:, :5], class_conf, class_pred), axis=1)
        detections = detections[conf_mask]

        if len(detections)<1:
            continue
        if class_agnostic:
            nms_out_index = nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre)
        else:
            nms_out_index = multiclass_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
                class_agnostic=class_agnostic)

        detections = detections[nms_out_index] # filtering boxes

        if output[i] is None:
            output[i] = detections
        else:
            output[i] = np.concatenate((output[i], detections))
    return output
