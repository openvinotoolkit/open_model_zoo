import numpy as np
from .preprocess import preprocess
from .postprocess import postprocess
from .settings import MwGlobalExp
from .deploy_util import demo_postprocess


class SubDetector:
    def __init__(self, exp: MwGlobalExp, backend: str='openvino'):
        self.inode, self.onode, self.input_shape, self.model = exp.get_openvino_model()

        self.cls_names = exp.mw_classes
        self.num_classes = exp.num_classes
        self.conf_thresh = exp.conf_thresh
        self.nms_thresh = exp.nms_thresh
        self.infer_request = self.model.create_infer_request()

    def inference(self, img):
        img_info = {"id": 0}
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width

        img_feed, ratio = preprocess(img, self.input_shape)
        img_info["ratio"] = ratio
        res = self.infer_request.infer({self.inode: np.expand_dims(img_feed, axis=0)})[self.onode]
        outputs = demo_postprocess(res, self.input_shape, p6=False)

        outputs = postprocess(
            outputs, self.num_classes, self.conf_thresh,
            self.nms_thresh, class_agnostic=True)

        return outputs, img_info

    def pseudolabel(self, output, img_info, idx_offset, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        image_id = img_info['id']
        if output is None:
            return img
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
