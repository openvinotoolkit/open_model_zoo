import os
import numpy as np

import cv2

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
    def __init__(self, input_height=384, input_width=288, output_height=48, output_width=36):
        self._num_keypoints = 17
        self.input_width = input_width
        self.input_height = input_height
        self.output_width = output_width
        self.output_height = output_height

    def __call__(self, img, bbox):
        c, s = preprocess_bbox(bbox, img)
        trans, _ = self.get_trasformation_matrix(c, s, [self.input_width, self.input_height])
        transformed_image = cv2.warpAffine(img, trans, (self.input_width, self.input_height), flags=cv2.INTER_LINEAR)
        rev_trans = self.get_trasformation_matrix(c, s, [self.output_width, self.output_height])[1]

        return rev_trans, transformed_image.transpose(2, 0, 1)[None, ]

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
    def __init__(self, ie, path_to_model_xml, scale=None, thr=-100, device='CPU'):
        self.model = ie.read_network(path_to_model_xml, os.path.splitext(path_to_model_xml)[0] + '.bin')

        assert len(self.model.input_info) == 1, "Expected 1 input blob"

        assert len(self.model.outputs) == 1, "Expected 1 output blob"

        self._input_layer_name = next(iter(self.model.input_info))
        self._output_layer_name = next(iter(self.model.outputs))
        self.CHANNELS_SIZE = 3
        self.OUTPUT_CHANNELS_SIZE = 17

        assert len(self.model.input_info[self._input_layer_name].input_data.shape) == 4 and \
               self.model.input_info[self._input_layer_name].input_data.shape[1] == self.CHANNELS_SIZE,\
               "Expected model input blob with shape [1, 3, H, W]"

        assert len(self.model.outputs[self._output_layer_name].shape) == 4 and \
               self.model.outputs[self._output_layer_name].shape[1] == self.OUTPUT_CHANNELS_SIZE,\
            "Expected model output shape [1, %s, H, W]" % (self.OUTPUT_CHANNELS_SIZE)

        self._ie = ie
        self._exec_model = self._ie.load_network(self.model, device)
        self._scale = scale
        self._thr = thr

        _, _, self.input_h, self.input_w = self.model.input_info[self._input_layer_name].input_data.shape
        _, _, self.output_h, self.output_w = self.model.outputs[self._output_layer_name].shape
        self._transform = TransformedCrop(self.input_h, self.input_w, self.output_h, self.output_w)
        self.infer_time = -1

    def _preprocess(self, img, bbox):
        return self._transform(img, bbox)

    def _infer(self, prep_img):
        t0 = cv2.getTickCount()
        output = self._exec_model.infer(inputs={self._input_layer_name: prep_img})
        self.infer_time = ((cv2.getTickCount() - t0) / cv2.getTickFrequency())
        return output[self._output_layer_name][0]

    @staticmethod
    def _postprocess(heatmaps, rev_trans):
        all_keypoints = [extract_keypoints(heatmap) for heatmap in heatmaps]
        all_keypoints_transformed = [affine_transform([kp[1][0], kp[1][1]], rev_trans) for kp in all_keypoints]

        return all_keypoints_transformed

    def estimate(self, img, bbox):
        rev_trans, preprocessed_img = self._preprocess(img, bbox)
        heatmaps = self._infer(preprocessed_img)
        keypoints = self._postprocess(heatmaps, rev_trans)
        return keypoints
