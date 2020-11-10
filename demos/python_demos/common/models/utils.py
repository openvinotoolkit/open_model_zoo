import cv2
import numpy as np


class Detection:
    def __init__(self, xmin, ymin, xmax, ymax, score, id):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.score = score
        self.id = id

    def bottom_left_point(self):
        return self.xmin, self.ymin

    def top_right_point(self):
        return self.xmax, self.ymax


class DetectionWithLandmarks(Detection):
    def __init__(self, xmin, ymin, xmax, ymax, score, id, landmarks_x, landmarks_y):
        super().__init__(xmin, ymin, xmax, ymax, score, id)
        self.landmarks = []
        for x, y in zip(landmarks_x, landmarks_y):
            self.landmarks.append((x, y))


def load_labels(label_file):
    with open(label_file, 'r') as f:
        labels_map = [x.strip() for x in f]
    return labels_map


def resize_image(image, size, keep_aspect_ratio=False):
    if not keep_aspect_ratio:
        resized_frame = cv2.resize(image, size)
    else:
        h, w = image.shape[:2]
        scale = min(size[1] / h, size[0] / w)
        resized_frame = cv2.resize(image, None, fx=scale, fy=scale)
    return resized_frame


def resize_image_letterbox(image, w, h)):
    ih, iw = image.shape[0:2]
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = cv2.resize(image, (nw, nh))
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    resized_image = np.pad(image, ((dy, dy + (h - nh) % 2), (dx, dx + (w - nw) % 2), (0, 0)),
                           mode='constant', constant_values=128)
    return resized_image
