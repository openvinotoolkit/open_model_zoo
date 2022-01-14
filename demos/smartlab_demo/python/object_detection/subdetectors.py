import torch
import numpy as np

from .preprocess import preprocess
from .postprocess import postprocess
from .settings import MwGlobalExp
from .deploy_util import demo_postprocess
from .vis import vis

class SubDetector(object):
    def __init__(self, exp:MwGlobalExp, backend:str='openvino'):
        self.inode, self.onode, self.input_shape, self.model = exp.get_openvino_model()

        self.cls_names = exp.mw_classes
        self.num_classes = exp.num_classes
        self.conf_thresh = exp.conf_thresh
        self.nms_thresh = exp.nms_thresh

    def inference(self, img):
        img_info = {"id": 0}
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width

        img_feed, ratio = preprocess(img, self.input_shape)
        img_info["ratio"] = ratio
        res = self.model.infer(inputs={self.inode:img_feed})[self.onode]
        outputs = demo_postprocess(res, self.input_shape, p6=False)

        outputs = postprocess(
            torch.tensor(outputs), self.num_classes, self.conf_thresh,
            self.nms_thresh, class_agnostic=True)

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
                output = np.cat(output)
                
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
