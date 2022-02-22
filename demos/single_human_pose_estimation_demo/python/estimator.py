import numpy as np
import cv2


def preprocess_bbox(bbox, image):
    aspect_ratio = 0.75
    bbox[0] = np.clip(bbox[0], 0, image.shape[1] - 1)
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


class TransformedCrop:
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


class HumanPoseEstimator:
    def __init__(self, core, model_path, device='CPU'):
        self.model = core.read_model(model_path)
        if len(self.model.inputs) != 1:
            raise RuntimeError("HumanPoseEstimator supports only models with 1 input layer")
        if len(self.model.outputs) != 1:
            raise RuntimeError("HumanPoseEstimator supports only models with 1 output layer")

        input_shape = self.model.inputs[0].shape
        if len(input_shape) != 4 or input_shape[1] != 3:
            raise RuntimeError("Expected model input shape [1, 3, H, W]")

        OUTPUT_CHANNELS_SIZE = 17
        output_shape = self.model.outputs[0].shape
        if len(output_shape) != 4 or output_shape[1] != OUTPUT_CHANNELS_SIZE:
            raise RuntimeError("Expected model output shape [1, {}, H, W]".format(OUTPUT_CHANNELS_SIZE))

        compiled_model = core.compile_model(self.model, device)
        self.output_tensor = compiled_model.outputs[0]
        self.infer_request = compiled_model.create_infer_request()
        self.input_tensor_name = self.model.inputs[0].get_any_name()
        self._transform = TransformedCrop(input_shape[2], input_shape[3], output_shape[2], output_shape[3])

    def _preprocess(self, img, bbox):
        return self._transform(img, bbox)

    def _infer(self, prep_img):
        input_data = {self.input_tensor_name: prep_img}
        output = self.infer_request.infer(input_data)[self.output_tensor]
        return output[0]

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
