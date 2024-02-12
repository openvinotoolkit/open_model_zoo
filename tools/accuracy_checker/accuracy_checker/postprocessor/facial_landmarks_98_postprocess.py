"""
Copyright (c) 2018-2024 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import cv2
from ..postprocessor.postprocessor import Postprocessor
from ..representation import FacialLandmarksHeatMapAnnotation, FacialLandmarksHeatMapPrediction


class Heatmap2Keypoints(Postprocessor):
    __provider__ = 'heatmap2keypoints'

    annotation_types = (FacialLandmarksHeatMapAnnotation, )
    prediction_types = (FacialLandmarksHeatMapPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        return parameters

    def configure(self):
        pass

    def process_image(self, annotation, prediction):
        for annotation_, prediction_ in zip(annotation, prediction):
            height, width, _ = self.image_size
            x_start, y_start = 0, 0
            resized_box = annotation_.metadata.get('rect')

            x_start, y_start, x_max, y_max = resized_box
            width = x_max - x_start
            height = y_max - y_start

            center, scale = self._xywh2cs(x_start, y_start, width, height)
            pred, _ = self._keypoints_from_heatmaps(np.array(prediction_.heatmap), [center], [scale])
            detLms_x = []
            detLms_y = []
            for pr in pred[0]:
                detLms_x.append(pr[0])
                detLms_y.append(pr[1])

            prediction_.x_values = detLms_x
            prediction_.y_values = detLms_y

        return annotation, prediction

    @classmethod
    def _xywh2cs(cls, x, y, w, h, padding=1.25):
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
        scale = scale * padding
        return center, scale

    @classmethod
    def _keypoints_from_heatmaps(cls, heatmaps, center, scale):

        def _get_max_preds(heatmaps):
            N, K, _, W = heatmaps.shape
            heatmaps_reshaped = heatmaps.reshape((N, K, -1))
            idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
            maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

            preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
            preds[:, :, 0] = preds[:, :, 0] % W
            preds[:, :, 1] = preds[:, :, 1] // W
            preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
            return preds, maxvals

        def _get_3rd_point(a, b):
            direction = a - b
            third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)
            return third_pt

        def rotate_point(pt, angle_rad):
            sn, cs = np.sin(angle_rad), np.cos(angle_rad)
            new_x = pt[0] * cs - pt[1] * sn
            new_y = pt[0] * sn + pt[1] * cs
            rotated_pt = [new_x, new_y]

            return rotated_pt

        def _get_affine_transform(center, scale, rot, output_size, shift=(0., 0.), inv=False):

            scale_tmp = scale * 200.0

            shift = np.array(shift)
            src_w = scale_tmp[0]
            dst_w = output_size[0]
            dst_h = output_size[1]

            rot_rad = np.pi * rot / 180
            src_dir = rotate_point([0., src_w * -0.5], rot_rad)
            dst_dir = np.array([0., dst_w * -0.5])

            src = np.zeros((3, 2), dtype=np.float32)
            src[0, :] = center + scale_tmp * shift
            src[1, :] = center + src_dir + scale_tmp * shift
            src[2, :] = _get_3rd_point(src[0, :], src[1, :])

            dst = np.zeros((3, 2), dtype=np.float32)
            dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
            dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
            dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

            if inv:
                trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
            else:
                trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

            return trans

        def _transform_preds(coords, center, scale, output_size, use_udp=False):
            target_coords = coords.copy()
            trans = _get_affine_transform(center, scale, 0, output_size, inv=True)
            for p in range(coords.shape[0]):
                target_coords[p, 0:2] = np.array(trans) @ np.array([coords[p, 0:2][0], coords[p, 0:2][1], 1.])
            return target_coords

        N, K, H, W = heatmaps.shape
        preds, maxvals = _get_max_preds(heatmaps)

        for n in range(N):
            for k in range(K):
                heatmap = heatmaps[n][k]
                px = int(preds[n][k][0])
                py = int(preds[n][k][1])
                if 1 < px < W - 1 and 1 < py < H - 1:
                    diff = np.array([
                        heatmap[py][px + 1] - heatmap[py][px - 1],
                        heatmap[py + 1][px] - heatmap[py - 1][px]
                    ])
                    preds[n][k] += np.sign(diff) * .25

        # Transform back to the image
        for i in range(N):
            preds[i] = _transform_preds(
                preds[i], center[i], scale[i], [W, H])

        return preds, maxvals
