import numpy as np

from .model import Model
from .utils import resize_image, DetectionWithLandmarks


class ProposalModel(Model):
    def __init__(self, ie, model_path):
        super().__init__(ie, model_path)
        self.image_blob_name = next(iter(self.net.input_info))
        for name, blob in self.net.outputs.items():
            if blob.shape[1] == 2:
                self.prob_blob_name = name
            elif blob.shape[1] == 4:
                self.bbox_blob_name = name

    def preprocess(self, inputs):
        # MTCNN input layout is NCWH
        image = inputs
        n, c, w, h = self.net.input_info[self.image_blob_name].input_data.shape
        resized_image = resize_image(image, (w, h))
        resized_image = resized_image.transpose((2, 1, 0))
        resized_image = resized_image.reshape((n, c, w, h))
        dict_inputs = {self.image_blob_name: resized_image}
        return dict_inputs, {}

    def postprocess(self, outputs, meta):
        bboxes = outputs[self.bbox_blob_name]
        scores = outputs[self.prob_blob_name]

        detections = self._generate_detections(bboxes[0], scores[0], 1/self.scale, 0.6)

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
            shift = 0.5 * (np.array([w, h]) - np.repeat([a], 2, axis=0)).T
            boxes[:, :2] += shift
            boxes[:, 2:4] -= shift
            return boxes

        out_side = max(bboxes.shape[1:])
        in_side = 2 * out_side + 11
        stride = 2
        if out_side != 1:
            stride = float(in_side - 12) / (out_side - 1)
        (x, y) = np.where(scores[1] >= score_threshold)
        if x.size == 0:
            return []

        kernels = np.array([x, y]).T
        kernels = np.concatenate((np.fix((stride * kernels) * scale),
                                  np.fix((stride * kernels + 11) * scale)),
                                 axis=1)
        offset = bboxes[:, x, y].T
        score = np.array([scores[1][x, y]]).T
        boxes = kernels + offset * 12.0 * scale
        boxes = np.concatenate((boxes, score), axis=1)
        boxes = square_box(boxes)
        result = []
        for box in boxes:
            if box[2] > box[0] and box[3] > box[1]:
                result.append(box)

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
            pr_scale = 1000.0/max(h, w)
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
    def NMS(bboxes, threshold, type):
        if len(bboxes)==0:
            return bboxes
        boxes = np.array(bboxes)
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


class RefineModel(Model):
    def __init__(self, ie, model_path):
        super().__init__(ie, model_path)
        self.image_blob_name = next(iter(self.net.input_info))
        for name, blob in self.net.outputs.items():
            if blob.shape[1] == 2:
                self.prob_blob_name = name
            elif blob.shape[1] == 4:
                self.bbox_blob_name = name
        self.n, self.c, self.w, self.h = self.net.input_info[self.image_blob_name].input_data.shape

    def preprocess(self, inputs):
        image = inputs['image']
        x1, y1, x2, y2 = [int(x) for x in inputs['crop']]
        image = image[y1:y2, x1:x2]
        resized_image = resize_image(image, (self.w, self.h))
        meta = {'crop': inputs['crop']}
        resized_image = resized_image.transpose((2, 1, 0))
        resized_image = resized_image.reshape((self.n, self.c, self.w, self.h))
        dict_inputs = {self.image_blob_name: resized_image}
        return dict_inputs, meta

    def postprocess(self, outputs, meta):
        bboxes = outputs[self.bbox_blob_name]
        scores = outputs[self.prob_blob_name]

        detections = self._generate_detections(bboxes[0], scores[0], meta['crop'], 0.7)

        return detections, meta

    def _generate_detections(self, bboxes, scores, crop, score_threshold):
        def square_box(boxes):
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            a = np.maximum(w, h)
            shift = 0.5 * (np.array([w, h]) - np.repeat([a], 2, axis=0)).T
            boxes[:, :2] += shift
            boxes[:, 2:4] -= shift
            return boxes
        if len(scores.shape) == 1:
            bboxes = np.expand_dims(bboxes, axis=1)
            scores = np.expand_dims(scores, axis=1)
        (keep,) = np.where(scores[1] >= score_threshold)
        if keep.size == 0:
            return []
        w = crop[2] - crop[0]
        h = crop[3] - crop[1]
        # bboxes = bboxes[:, keep]
        bboxes[0] = bboxes[0] * w + crop[0]
        bboxes[1] = bboxes[1] * w + crop[1]
        bboxes[2] = bboxes[2] * h + crop[2]
        bboxes[3] = bboxes[3] * h + crop[3]
        score = np.array([scores[1]]).T
        boxes = np.concatenate((bboxes.T, score), axis=1)
        boxes = square_box(boxes)
        result = []
        for box in boxes:
            if box[2] > box[0] and box[3] > box[1]:
                result.append(box)

        return ProposalModel.NMS(result, 0.5, 'iou')

    @staticmethod
    def postprocess_all(detections):
        detections = ProposalModel.NMS(detections, 0.7, 'iou')
        detections = [Detection(*detection, 1) for detection in detections]
        return detections


class OutputModel(Model):
    def __init__(self, ie, model_path):
        super().__init__(ie, model_path)
        self.image_blob_name = next(iter(self.net.input_info))
        for name, blob in self.net.outputs.items():
            if blob.shape[1] == 2:
                self.prob_blob_name = name
            elif blob.shape[1] == 4:
                self.bbox_blob_name = name
            elif blob.shape[1] == 10:
                self.landmarks_blob_name = name
        self.n, self.c, self.w, self.h = self.net.input_info[self.image_blob_name].input_data.shape

    def preprocess(self, inputs):
        image = inputs['image']
        x1, y1, x2, y2 = [int(x) for x in inputs['crop']]
        image = image[y1:y2, x1:x2]
        resized_image = resize_image(image, (self.w, self.h))
        meta = {'crop': inputs['crop']}
        resized_image = resized_image.transpose((2, 1, 0))
        resized_image = resized_image.reshape((self.n, self.c, self.w, self.h))
        dict_inputs = {self.image_blob_name: resized_image}
        return dict_inputs, meta

    def postprocess(self, outputs, meta):
        bboxes = outputs[self.bbox_blob_name]
        scores = outputs[self.prob_blob_name]
        landmarks = outputs[self.landmarks_blob_name]

        detections = self._generate_detections(bboxes[0], scores[0], landmarks[0], meta['crop'], 0.7)

        return detections, meta

    def _generate_detections(self, bboxes, scores, landmarks, crop, score_threshold):
        if len(scores.shape) == 1:
            bboxes = np.expand_dims(bboxes, axis=1)
            scores = np.expand_dims(scores, axis=1)
            landmarks = np.expand_dims(landmarks, axis=1)
        (keep,) = np.where(scores[1] >= score_threshold)
        if keep.size == 0:
            return []
        w = crop[2] - crop[0]
        h = crop[3] - crop[1]
        bboxes = bboxes[:, keep]
        bboxes[0] = bboxes[0] * w + crop[0]
        bboxes[1] = bboxes[1] * w + crop[1]
        bboxes[2] = bboxes[2] * h + crop[2]
        bboxes[3] = bboxes[3] * h + crop[3]
        landmarks[:5] = landmarks[:5] * w + crop[0]
        landmarks[5:] = landmarks[5:] * h + crop[1]
        # landmarks = np.concatenate((landmarks[0::2], landmarks[1::2]), axis=0)
        score = np.array([scores[1]]).T
        classes = np.ones(shape=(score.shape[0], 1))
        boxes = np.concatenate((bboxes.T, score, classes, landmarks.T), axis=1)
        result = []
        for box in boxes:
            if box[2] > box[0] and box[3] > box[1]:
                result.append(box)

        return ProposalModel.NMS(result, 0.5, 'iou')

    @staticmethod
    def postprocess_all(detections):
        detections = ProposalModel.NMS(detections, 0.7, 'iou')
        detections = [DetectionWithLandmarks(*detection[:6], detection[6:11], detection[11:])
                      for detection in detections]
        return detections
