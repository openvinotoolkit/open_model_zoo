import os
import numpy as np
from accessify import private

import cv2
from openvino.inference_engine import IENetwork

def preprocess_bbox(bbox, image):
    aspect_ratio = 0.75
    bbox[0] = np.clip(bbox[0], 0, image.shape[0] - 1)
    bbox[1] = np.clip(bbox[1], 0, image.shape[0] - 1)
    x2 = np.min((image.shape[1] - 1, bbox[0] + np.max((0, bbox[2] - 1))))
    y2 = np.min((image.shape[0] - 1, bbox[1] + np.max((0, bbox[3] - 1))))

    bbox = [bbox[0], bbox[1], x2 - bbox[0], y2 - bbox[1]]

    cx_bbox = bbox[0] + bbox[2] * 0.5
    cy_bbox = bbox[1] + bbox[3] * 0.5
    center = np.array([np.float32(cx_bbox), np.float32(cy_bbox)])

    if bbox[2] > aspect_ratio * bbox[3]:
        bbox[3] = bbox[2] * 1.0 / aspect_ratio
    elif bbox[2] < aspect_ratio * bbox[3]:
        bbox[2] = bbox[3] * aspect_ratio

    s = np.array([bbox[2], bbox[3]], np.float32)
    scale = s * 1.25

    return center, scale


def extract_keypoints(heatmap, min_confidence=-100):
    ind = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)
    if heatmap[ind] < min_confidence:
        ind = (-1, -1)
    else:
        ind = (int(ind[1]), int(ind[0]))
    return heatmap[ind[1]][ind[0]], ind


def affine_transform(pt, t):
        transformed_point = np.dot(t, [pt[0], pt[1], 1.])[:2]
        return transformed_point


class TransformedCrop(object):
    def __init__(self, input_weight=288, input_height=384):
        self._num_keypoints = 17
        self._weight = input_weight
        self._height = input_height

    def __call__(self, img, bbox):
        c, s = preprocess_bbox(bbox, img)
        trans, _ = self.get_trasformation_matrix(c, s, [self._weight, self._height])
        transformed_image = cv2.warpAffine(img, trans, (self._weight, self._height), flags=cv2.INTER_LINEAR)
        rev_trans = self.get_trasformation_matrix(c, s, [36, 48])[1]

        return trans,  rev_trans, transformed_image.transpose(2, 0, 1)[None, ]

    @staticmethod
    def get_trasformation_matrix(center, scale, output_size):

        w, h = scale
        points = np.zeros((3, 2), dtype=np.float32)
        transformed_points = np.zeros((3, 2), dtype=np.float32)

        transformed_points[0, :] = [output_size[0] * 0.5, output_size[1] * 0.5]
        transformed_points[1, :] = [output_size[0] * 0.5, output_size[1] * 0.5 - output_size[0] * 0.5]
        transformed_points[2, :] = [0, output_size[1] * 0.5]

        shift_y = [0, - w * 0.5]
        shift_x = [- w * 0.5, 0]

        points[0, :] = center
        points[1, :] = center + shift_y
        points[2, :] = center + shift_x

        rev_trans = cv2.getAffineTransform(np.float32(transformed_points), np.float32(points))

        trans = cv2.getAffineTransform(np.float32(points), np.float32(transformed_points))

        return trans, rev_trans


class HumanPoseEstimator(object):
    def __init__(self, ie, path_to_model_xml, path_to_lib, scale=None, thr=-100, device='CPU'):
        self.model = IENetwork(model=path_to_model_xml, weights=os.path.splitext(path_to_model_xml)[0] + '.bin')
        self._input_layer_name = next(iter(self.model.inputs))
        self._output_layer_name = next(iter(self.model.outputs))

        assert len(self.model.outputs) == 1, "Expected 1 output blob"

        assert len(self.model.outputs[self._output_layer_name].shape) == 4, "Expected model output shape [1, N, H, W]"

        self._ie = ie
        if device == 'CPU':
            self._ie.add_extension(path_to_lib, device)
        self._exec_model = self._ie.load_network(self.model, device)
        self._scale = scale
        self._thr = thr

        _, _, self.input_h, self.input_w = self.model.inputs[self._input_layer_name].shape
        self._transform = TransformedCrop()
        self.infer_time = -1

    @private
    def preprocess(self, img, bbox):
        return self._transform(img, bbox)

    @private
    def infer(self, prep_img):
        t0 = cv2.getTickCount()
        output = self._exec_model.infer(inputs={self._input_layer_name: prep_img})
        self.infer_time = ((cv2.getTickCount() - t0) / cv2.getTickFrequency())
        return output[self._output_layer_name][0]

    @staticmethod
    def postprocess(heatmaps, rev_trans):
        all_keypoints = [extract_keypoints(heatmap) for heatmap in heatmaps]
        all_keypoints_transformed = [affine_transform([kp[1][0], kp[1][1]], rev_trans) for kp in all_keypoints]

        return all_keypoints_transformed

    def estimate(self, img, bbox):
        trans, rev_trans, preprocessed_img = self.preprocess(img, bbox)
        heatmaps = self.infer(preprocessed_img)
        keypoints = self.postprocess(heatmaps, rev_trans)
        return keypoints
