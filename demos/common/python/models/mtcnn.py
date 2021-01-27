import numpy as np

from .model import Model
from .utils import resize_image, Detection


class ProposalModel(Model):
    def __init__(self, ie, model_path):
        super().__init__(ie, model_path)
        self.image_blob_name = next(iter(self.net.input_info))
        for name, blob in self.net.outputs.items():
            if blob.shape[1] == 2:
                self.prob_blob_name = name
            elif blob.shape[1] == 4:
                self.bbox_blob_name = name

        # self.n, self.c, self.h, self.w = self.net.input_info[self.image_blob_name].input_data.shape

    def preprocess(self, inputs):
        # MTCNN input layout is NCWH
        image = inputs
        n, c, w, h = self.net.input_info[self.image_blob_name].input_data.shape
        resized_image = resize_image(image, (w, h))
        meta = {'original_shape': image.shape,
                'resized_shape': resized_image.shape}
        resized_image = resized_image.transpose((2, 1, 0))
        resized_image = resized_image.reshape((n, c, w, h))
        dict_inputs = {self.image_blob_name: resized_image}
        return dict_inputs, meta

    def postprocess(self, outputs, meta):
        bboxes = outputs[self.bbox_blob_name]
        scores = outputs[self.prob_blob_name]
        orginal_image_shape = meta['original_shape']
        resized_image_shape = meta['resized_shape']

        detections = self._generate_detections(bboxes[0], scores[0], 1/self.scale, 0.6)
        # detections = [Detection(*detection, 1) for detection in detections]

        return detections, meta

    @staticmethod
    def postprocess_all(detections):
        detections = ProposalModel.NMS(detections, 0.7, 'iou')
        detections = [Detection(*detection, 1) for detection in detections]
        return detections

    @staticmethod
    def _generate_detections(bboxes, scores, scale, score_threshold):
        def square_box(boxes):
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            a = np.maximum(w, h)
            shift = 0.5 * (np.array([w, h]) - np.repeat([a], 2, axis=0))
            boxes[:, 0] += shift[0]
            boxes[:, 1] += shift[1]
            boxes[:, 2] -= shift[0]
            boxes[:, 3] -= shift[1]
            return boxes
        out_side = max(bboxes.shape[1:])
        in_side = 2 * out_side + 11
        stride = 2
        if out_side != 1:
            stride = float(in_side - 12) / (out_side - 1)
        (x, y) = np.where(scores[1] >= score_threshold)
        if x.size == 0:
            return []
        boundingbox = np.array([x, y]).T
        bb1 = np.fix((stride * boundingbox) * scale)
        bb2 = np.fix((stride * boundingbox + 11) * scale)
        boundingbox = np.concatenate((bb1, bb2), axis=1)
        dx1 = bboxes[0][x, y]
        dx2 = bboxes[1][x, y]
        dx3 = bboxes[2][x, y]
        dx4 = bboxes[3][x, y]
        score = np.array([scores[1][x, y]]).T
        offset = np.array([dx1, dx2, dx3, dx4]).T
        boundingbox = boundingbox + offset * 12.0 * scale
        rectangles = np.concatenate((boundingbox, score), axis=1)
        rectangles = square_box(rectangles)
        result = []
        for rectangle in rectangles:
            if rectangle[2] > rectangle[0] and rectangle[3] > rectangle[1]:
                result.append(rectangle)

        return ProposalModel.NMS(result, 0.5, 'iou')

    @staticmethod
    def calc_scales(image):
        pr_scale = 1.0
        h, w = image.shape[:2]
        if min(w, h) > 1000:
            pr_scale = 1000.0/min(h, w)
            w = int(w * pr_scale)
            h = int(h * pr_scale)
        elif max(w, h) < 1000:
            scale = 1000.0/max(h, w)
            w = int(w * pr_scale)
            h = int(h * pr_scale)

        #multi-scale
        scales = []
        factor = 0.709
        factor_count = 0
        minl = min(h, w)
        while minl >= 12:
            scales.append(pr_scale*pow(factor, factor_count))
            minl *= factor
            factor_count += 1
        return scales

    @staticmethod
    def NMS(rectangles,threshold,type):
        if len(rectangles)==0:
            return rectangles
        boxes = np.array(rectangles)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        s  = boxes[:, 4]
        area = np.multiply(x2-x1+1, y2-y1+1)
        I = np.array(s.argsort())
        pick = []
        while len(I)>0:
            xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]]) #I[-1] have hightest prob score, I[0:-1]->others
            yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
            xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
            yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            if type == 'iom':
                o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
            else:
                o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
            pick.append(I[-1])
            I = I[np.where(o<=threshold)[0]]
        result_rectangle = boxes[pick].tolist()
        return result_rectangle
