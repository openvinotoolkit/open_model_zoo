import random
import numpy as np
import colorsys
import cv2
import os
import sys
import urllib

open_model_zoo_path =  os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(os.curdir))))

sys.path.append(os.path.join(open_model_zoo_path, "demos", "common", "python"))

from models import SSD, YOLO, FaceBoxes, CenterNet, RetinaFace

class ColorPalette:
    def __init__(self, n, rng=None):
        assert n > 0

        if rng is None:
            rng = random.Random(0xACE)

        candidates_num = 100
        hsv_colors = [(1.0, 1.0, 1.0)]
        for _ in range(1, n):
            colors_candidates = [(rng.random(), rng.uniform(0.8, 1.0), rng.uniform(0.5, 1.0))
                                 for _ in range(candidates_num)]
            min_distances = [self.min_distance(hsv_colors, c) for c in colors_candidates]
            arg_max = np.argmax(min_distances)
            hsv_colors.append(colors_candidates[arg_max])

        self.palette = [self.hsv2rgb(*hsv) for hsv in hsv_colors]

    @staticmethod
    def dist(c1, c2):
        dh = min(abs(c1[0] - c2[0]), 1 - abs(c1[0] - c2[0])) * 2
        ds = abs(c1[1] - c2[1])
        dv = abs(c1[2] - c2[2])
        return dh * dh + ds * ds + dv * dv

    @classmethod
    def min_distance(cls, colors_set, color_candidate):
        distances = [cls.dist(o, color_candidate) for o in colors_set]
        return np.min(distances)

    @staticmethod
    def hsv2rgb(h, s, v):
        return tuple(round(c * 255) for c in colorsys.hsv_to_rgb(h, s, v))

    def __getitem__(self, n):
        return self.palette[n % len(self.palette)]

    def __len__(self):
        return len(self.palette)


def get_model(ie, model, architecture_type, labels, keep_aspect_ratio=False, prob_threshold=0.5):
    if architecture_type == "ssd":
        return SSD(ie, model, labels=labels, keep_aspect_ratio_resize=keep_aspect_ratio)
    elif architecture_type == "yolo":
        return YOLO(ie, model, labels=labels, threshold=prob_threshold, keep_aspect_ratio=keep_aspect_ratio)
    elif architecture_type == "faceboxes":
        return FaceBoxes(ie, model, threshold=prob_threshold)
    elif architecture_type == "centernet":
        return CenterNet(ie, model, labels=labels, threshold=prob_threshold)
    elif architecture_type == "retina":
        return RetinaFace(ie, model, threshold=prob_threshold)
    else:
        raise RuntimeError("No model type or invalid model type (-at) provided: {}".format(architecture_type))


def put_highlighted_text(frame, message, position, font_face, font_scale, color, thickness):
    cv2.putText(frame, message, position, font_face, font_scale, (255, 255, 255), thickness + 1)  # white border
    cv2.putText(frame, message, position, font_face, font_scale, color, thickness)


def draw_detections(frame, detections, palette, labels, threshold, draw_landmarks=False):
    """
    Draw detection boxes on `frame`.
    """
    size = frame.shape[:2]
    for detection in detections:
        if detection.score > threshold:
            xmin = max(int(detection.xmin), 0)
            ymin = max(int(detection.ymin), 0)
            xmax = min(int(detection.xmax), size[1])
            ymax = min(int(detection.ymax), size[0])
            class_id = int(detection.id)
            color = palette[class_id]
            det_label = labels[class_id] if labels and len(labels) >= class_id else "#{}".format(class_id)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(
                frame,
                "{} {:.1%}".format(det_label, detection.score),
                (xmin, ymin - 7),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                color,
                1,
            )
            if draw_landmarks:
                for landmark in detection.landmarks:
                    cv2.circle(frame, landmark, 2, (0, 255, 255), 2)
    return frame


def download_video(path: str) -> np.ndarray:
    """
    Download a video from `path` and save it to the current folder.
    """

    # Set User-Agent to Mozilla because some websites block requests with User-Agent Python
    request = urllib.request.Request(path, headers={"User-Agent": "Mozilla/5.0"})
    response = urllib.request.urlopen(request)
    data = response.read()
    with open(os.path.basename(path), "wb") as f:
        f.write(data)
