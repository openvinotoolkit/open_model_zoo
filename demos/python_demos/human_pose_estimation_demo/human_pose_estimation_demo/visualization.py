import colorsys
import random

import cv2
import numpy as np


default_skeleton = ((16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13), (6, 7),
    (6, 8), (7, 9), (8, 10), (9, 11), (2, 3), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7))


def show_poses(img, poses, scores, threshold=0.5, skeleton=default_skeleton):
    if poses.size == 0:
        return img

    colors = [[0, 113, 188],
        [216, 82, 24],
        [236, 176, 31],
        [125, 46, 141],
        [118, 171, 47],
        [76, 189, 237],
        [161, 19, 46],
        [76, 76, 76],
        [153, 153, 153],
        [255, 0, 0],
        [255, 127, 0],
        [190, 190, 0],
        [0, 255, 0],
        [0, 0, 255],
        [170, 0, 255]]
            
    for idx, kp in enumerate(poses):
        points = [(int(p[0]), int(p[1])) for p in kp]
        is_visible = [p[2] for p in kp]
        if skeleton is not None:
            for bone in skeleton:
                i = bone[0] - 1
                j = bone[1] - 1
                if is_visible[i] > threshold and is_visible[j] > threshold:
                    cv2.line(img, points[i], points[j], thickness=2, color=colors[idx % len(colors)])
        for p, v in zip(points, is_visible):
            if v:
                cv2.circle(img, p, 1, (0, 0, 255), 2)
    return img
