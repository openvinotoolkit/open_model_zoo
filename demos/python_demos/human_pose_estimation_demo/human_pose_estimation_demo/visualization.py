import cv2
import numpy as np


default_skeleton = ((16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13), (6, 7),
    (6, 8), (7, 9), (8, 10), (9, 11), (2, 3), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7))


def show_poses(img, poses, scores, pose_score_threshold=0.5, point_score_threshold=0.5, skeleton=None):
    if poses.size == 0:
        return img

    if skeleton is None:
        skeleton = default_skeleton

    colors = (
        (255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85),
        (255, 0, 170), (85, 255, 0), (255, 170, 0), (0, 255, 0),
        (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
        (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255),
        (0, 170, 255))

    stick_width = 4

    for idx, (pose, pose_score) in enumerate(zip(poses, scores)):
        if pose_score <= pose_score_threshold:
            continue
        points = pose[:, :2].astype(int)
        points_scores = pose[:, 2]
        for i, (p, v) in enumerate(zip(points, points_scores)):
            if v > point_score_threshold:
                cv2.circle(img, tuple(p), 1, colors[i], 2)

    img_x = np.copy(img)
    for idx, (pose, pose_score) in enumerate(zip(poses, scores)):
        if pose_score <= pose_score_threshold:
            continue
        points = pose[:, :2].astype(int)
        points_scores = pose[:, 2]
        for bone in skeleton:
            i = bone[0] - 1
            j = bone[1] - 1
            if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                middle = (points[i] + points[j]) // 2
                vec = points[i] - points[j]
                length = np.sqrt((vec * vec).sum())
                angle = int(np.arctan2(vec[1], vec[0]) * 180 / np.pi)
                polygon = cv2.ellipse2Poly(tuple(middle), (int(length / 2), min(int(length / 50), stick_width)), angle, 0, 360, 1)
                cv2.fillConvexPoly(img_x, polygon, colors[j])
    cv2.addWeighted(img, 0.4, img_x, 0.6, 0, dst=img)
        
    return img
