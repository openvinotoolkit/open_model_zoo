import math
from operator import itemgetter

import numpy as np
import torch


# BODY_PARTS_KPT_IDS = tuple((i - 1, j - 1) for i, j in ((1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
#                       (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)))
# # BODY_PARTS_KPT_IDS = ((1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
# #                       (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17))
# BODY_PARTS_PAF_IDS = ((12, 13), (20, 21), (14, 15), (16, 17), (22, 23), (24, 25), (0, 1), (2, 3), (4, 5),
#                       (6, 7), (8, 9), (10, 11), (28, 29), (30, 31), (34, 35), (32, 33), (36, 37), (18, 19), (26, 27))


BODY_PARTS_KPT_IDS = tuple((i - 1, j - 1) for i, j in 
    [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13],
    [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18], [3, 17], [6, 18]]
)
BODY_PARTS_PAF_IDS = tuple(i - 19 for i, _ in [
    [31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], [23, 24], [25, 26],
    [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], [55, 56], [37, 38], [45, 46]
])

class OpenPoseDecoder:
    def __init__(self, num_joints, max_points=100, score_threshold=0.1, min_dist=6.0, delta=0.5, out_stride=8):
        super().__init__()
        self.num_joints = num_joints
        self.max_points = max_points
        self.score_threshold = score_threshold
        self.min_dist = min_dist
        self.delta = delta
        self.out_stride = out_stride
        self.high_res_heatmaps = False
        self.high_res_pafs = False
        self.arange_cache = {10: np.arange(10)}
        nms_kernel = 5
        self.pool = torch.nn.MaxPool2d(nms_kernel, 1, (nms_kernel - 1) // 2)

    def nms(self, heatmaps):
        heatmaps = torch.as_tensor(heatmaps)
        maxm = self.pool(heatmaps)
        maxm = torch.eq(maxm, heatmaps).float()
        return (heatmaps * maxm).numpy()

    def scale_kpts(self, kpts, scale, max_v):
        for kpt in kpts:
            kpt[0] = max(0, min(kpt[0] * scale, max_v[0]))
            kpt[1] = max(0, min(kpt[1] * scale, max_v[1]))
        return kpts

    def __call__(self, heatmaps, nms_heatmaps, pafs):
        pafs = np.transpose(pafs, (0, 2, 3, 1))

        batch_size, _, h, w = heatmaps.shape
        hup = h * self.out_stride
        wup = w * self.out_stride
        assert batch_size == 1

        min_dist = self.min_dist
        if not self.high_res_heatmaps:
            self.min_dist = 6.0 / self.out_stride

        keypoints = self.extract_points(heatmaps, nms_heatmaps)
        self.min_dist = min_dist

        if self.high_res_heatmaps:
            for kpts in keypoints:
                self.scale_kpts(kpts, 1 / self.out_stride, (w - 1, h - 1))

        if self.delta > 0:
            # To adjust coordinates' flooring in heatmaps target generation.
            for kpts in keypoints:
                for kpt in kpts:
                    kpt[0] = min(kpt[0] + self.delta, w - 1)
                    kpt[1] = min(kpt[1] + self.delta, h - 1)

        if self.high_res_pafs:
            for kpts in keypoints:
                self.scale_kpts(kpts, self.out_stride, (wup - 1, hup - 1))

        pose_entries, keypoints = self.group_keypoints(keypoints, pafs, pose_entry_size=self.num_joints + 2)

        grouped_kpts, scores = self.convert_to_coco_format(pose_entries, keypoints, None)
        if len(grouped_kpts) > 0:
            grouped_kpts = np.asarray(grouped_kpts)
            grouped_kpts = grouped_kpts.reshape((grouped_kpts.shape[0], -1, 3))
            if not self.high_res_pafs:
                grouped_kpts[:, :, :2] *= self.out_stride
                grouped_kpts[:, :, 0].clip(0, wup - 1)
                grouped_kpts[:, :, 1].clip(0, hup - 1)
        else:
            grouped_kpts = np.empty((0, 17, 3), dtype=np.float32)
            scores = np.zeros(0, dtype=np.float32)

        return grouped_kpts, scores
        
    def extract_points(self, heatmaps, nms_heatmaps):
        batch_size, channels_num, h, w = heatmaps.shape
        assert batch_size == 1, 'Batch size of 1 only supported'
        assert channels_num >= self.num_joints

        xs, ys, scores = self.top_k(nms_heatmaps)

        masks = scores > self.score_threshold
        all_keypoints = []
        keypoint_id = 0
        for k in range(self.num_joints):
            # Filter low-score points.
            mask = masks[0, k]
            x = xs[0, k][mask].ravel()
            y = ys[0, k][mask].ravel()
            score = scores[0, k][mask].ravel()
            n = len(x)

            if n == 0:
                all_keypoints.append([])
                continue

            # Second stage of NMS.
            xx = x[:, None] - x[None, :]
            yy = y[:, None] - y[None, :]
            dists = np.sqrt(xx * xx + yy * yy)
            dists += np.tri(n, n, dtype=np.float32) * (2 * self.min_dist)
            keep = dists.min(axis=0) >= self.min_dist
            x = x[keep]
            y = y[keep]
            score = score[keep]

            # Apply quarter offset to improve localization accuracy.
            x, y = self.refine(heatmaps[0, k], x, y)
            np.core.umath.clip(x, 0, w - 1, out=x)
            np.core.umath.clip(y, 0, h - 1, out=y)

            # Pack resulting points.
            # keypoints = []
            # for a, b, c in zip(x, y, score):
            #     keypoints.append([a, b, c, keypoint_id])
            #     keypoint_id += 1
            n = len(x)
            keypoints = np.empty((n, 4), dtype=np.float32)
            keypoints[:, 0] = x
            keypoints[:, 1] = y
            keypoints[:, 2] = score
            keypoints[:, 3] = np.arange(keypoint_id, keypoint_id + n)
            # keypoints = np.stack([x, y, score, np.arange(keypoint_id, keypoint_id + n)], axis=1).astype(dtype=np.float32)
            keypoint_id += n

            all_keypoints.append(keypoints)
        return all_keypoints

    def top_k(self, heatmaps):
        N, K, _, W = heatmaps.shape
        heatmaps = heatmaps.reshape(N, K, -1)
        ind = heatmaps.argpartition(-self.max_points, axis=2)[:, :, -self.max_points:]
        scores = np.take_along_axis(heatmaps, ind, axis=2)
        subind = np.argsort(-scores, axis=2)
        ind = np.take_along_axis(ind, subind, axis=2)
        scores = np.take_along_axis(scores, subind, axis=2)
        y, x = np.divmod(ind, W)
        return x, y, scores

    def refine(self, heatmap, x, y):
        h, w = heatmap.shape[-2:]
        valid = np.logical_and(np.logical_and(0 < x, x < w - 1), np.logical_and(0 < y, y < h - 1))
        xx = x[valid]
        yy = y[valid]
        dx = np.sign(heatmap[yy, xx + 1] - heatmap[yy, xx - 1], dtype=np.float32) * 0.25
        dy = np.sign(heatmap[yy + 1, xx] - heatmap[yy - 1, xx], dtype=np.float32) * 0.25
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        x[valid] += dx
        y[valid] += dy
        return x, y
      
    

    def group_keypoints(self, all_keypoints_by_type, pafs, pose_entry_size=20, min_paf_score=0.05,
                        skeleton=BODY_PARTS_KPT_IDS, bones_to_channels=BODY_PARTS_PAF_IDS):

        all_keypoints = np.asarray([item for sublist in all_keypoints_by_type for item in sublist], dtype=np.float32)
        pose_entries = []

        point_num = 10
        grid = np.arange(point_num, dtype=np.float32).reshape(1, -1, 1)
            
        for part_id, paf_channel in enumerate(bones_to_channels):
            part_pafs = pafs[0, :, :, paf_channel:paf_channel + 2]

            kpt_a_id, kpt_b_id = skeleton[part_id]
            kpts_a = all_keypoints_by_type[kpt_a_id]
            kpts_b = all_keypoints_by_type[kpt_b_id]
            num_kpts_a = len(kpts_a)
            num_kpts_b = len(kpts_b)
            
            if num_kpts_a == 0 or num_kpts_b == 0:
                continue
            
            a = np.asarray([p[0:2] for p in kpts_a], dtype=np.float32)
            b = np.asarray([p[0:2] for p in kpts_b], dtype=np.float32)
            n, m = len(a), len(b)

            a = np.broadcast_to(a[None], (m, n, 2))
            vec_raw = (b[:, None, :] - a).reshape(-1, 1, 2)

            vec_norm = np.linalg.norm(vec_raw, ord=2, axis=-1, keepdims=True)
            vec = vec_raw / (vec_norm + 1e-6)
            steps = (1 / (point_num - 1) * vec_raw)
            points = steps * grid + a.reshape(-1, 1, 2)
            points = points.round().astype(dtype=np.int32)

            x = points[..., 0].ravel()
            y = points[..., 1].ravel()
            field = part_pafs[y, x].reshape(-1, point_num, 2)
            dot_prod = (field * vec).sum(-1).reshape(-1, point_num)

            valid_prod = dot_prod > min_paf_score
            valid_num = valid_prod.sum(1)
            success_ratio = valid_num / point_num
            score = (dot_prod * valid_prod).sum(1) / (valid_num + 1e-6)

            valid_limbs = np.where(np.logical_and(score > 0, success_ratio > 0.8))[0]
            b_idx, a_idx = np.divmod(valid_limbs, n)
            connections = []
            for t, i, j in zip(valid_limbs, a_idx, b_idx):
                connections.append([i, j, score[t], score[t] + kpts_a[i][2] + kpts_b[j][2]])

            if len(connections) > 0:
                connections = sorted(connections, key=itemgetter(2), reverse=True)

            num_connections = min(num_kpts_a, num_kpts_b)
            has_kpt_a = np.zeros(num_kpts_a, dtype=np.int32)
            has_kpt_b = np.zeros(num_kpts_b, dtype=np.int32)
            filtered_connections = []
            for row in range(len(connections)):
                if len(filtered_connections) == num_connections:
                    break
                i, j, cur_point_score = connections[row][0:3]
                if not has_kpt_a[i] and not has_kpt_b[j]:
                    filtered_connections.append([int(kpts_a[i][3]), int(kpts_b[j][3]), cur_point_score])
                    has_kpt_a[i] = 1
                    has_kpt_b[j] = 1
            connections = filtered_connections
            if len(connections) == 0:
                continue

            # for i in range(len(connections)):
            #     pose_entries.append(init_pose(kpt_a_id, kpt_b_id, connections[i]))
            # continue

            if part_id == 0:
                pose_entries = [np.full(pose_entry_size, -1) for _ in range(len(connections))]
                for i in range(len(connections)):
                    pose_entries[i][kpt_a_id] = connections[i][0]
                    pose_entries[i][kpt_b_id] = connections[i][1]
                    pose_entries[i][-1] = 2
                    # pose score = sum of all points' scores + sum of all connections' scores
                    pose_entries[i][-2] = np.sum(all_keypoints[connections[i][0:2], 2]) + connections[i][2]
            else:
                for connection in connections:
                    pose_a_idx = -1
                    pose_b_idx = -1
                    for j, pose in enumerate(pose_entries):
                        if pose[kpt_a_id] == connection[0]:
                            pose_a_idx = j
                        if pose[kpt_b_id] == connection[1]:
                            pose_b_idx = j
                    if pose_a_idx < 0 and pose_b_idx < 0:
                        # print('CREATE NEW POSE')
                        # Create new pose entry.
                        pose_entry = np.full(pose_entry_size, -1)
                        pose_entry[kpt_a_id] = connection[0]
                        pose_entry[kpt_b_id] = connection[1]
                        pose_entry[-1] = 2
                        pose_entry[-2] = np.sum(all_keypoints[connection[0:2], 2]) + connection[2]
                        pose_entries.append(pose_entry)
                    elif pose_a_idx >= 0 and pose_b_idx >= 0 and pose_a_idx != pose_b_idx:
                        # Merge two disjoint components into one pose.
                        pose_a = pose_entries[pose_a_idx]
                        pose_b = pose_entries[pose_b_idx]
                        do_merge_poses = True
                        for j in range(len(pose_b) - 2):
                            if pose_a[j] >= 0 and pose_b[j] >= 0 and pose_a[j] != pose_b[j]:
                                do_merge_poses = False
                                break
                        if not do_merge_poses:
                            continue
                        for j in range(len(pose_b) - 2):
                            if pose_b[j] >= 0:
                                pose_a[j] = pose_b[j]
                        # pose_a[kpt_b_id] = connection[1]
                        pose_a[-1] += pose_b[-1]
                        pose_a[-2] += pose_b[-2] + connection[2]
                        del pose_entries[pose_b_idx]
                    elif pose_a_idx >= 0:
                        # Add a new bone into pose.
                        pose = pose_entries[pose_a_idx]
                        if pose[kpt_b_id] < 0:
                            pose[-2] += all_keypoints[connection[1], 2]
                        pose[kpt_b_id] = connection[1]
                        pose[-2] += connection[2]
                        pose[-1] += 1
                    elif pose_b_idx >= 0:
                        # Add a new bone into pose.
                        pose = pose_entries[pose_b_idx]
                        if pose[kpt_a_id] < 0:
                            pose[-2] += all_keypoints[connection[0], 2]
                        pose[kpt_a_id] = connection[0]
                        pose[-2] += connection[2]
                        pose[-1] += 1

        filtered_entries = []
        for i in range(len(pose_entries)):
            if pose_entries[i][-1] < 3:  # or (pose_entries[i][-2] / pose_entries[i][-1] < 0.2):
                continue
            filtered_entries.append(pose_entries[i])
        pose_entries = np.asarray(filtered_entries)
        # pose_entries = np.asarray(pose_entries)
        return pose_entries, all_keypoints

    def convert_to_coco_format(self, pose_entries, all_keypoints, reorder_map=None):
        num_joints = 17
        coco_keypoints = []
        scores = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            keypoints = np.zeros(num_joints * 3)
            # if reorder_map is None:
            #     reorder_map = tuple(range(num_joints))
            reorder_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
            person_score = pose_entries[n][-2]
            for keypoint_id, target_id in zip(pose_entries[n][:-2], reorder_map):
                if target_id < 0:
                    continue
                cx, cy, score, visibility = 0, 0, 0, 0  # keypoint not found
                if keypoint_id != -1:
                    cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                    visibility = 2
                keypoints[target_id * 3 + 0] = cx
                keypoints[target_id * 3 + 1] = cy
                keypoints[target_id * 3 + 2] = score
            coco_keypoints.append(keypoints)
            scores.append(person_score * max(0, (pose_entries[n][-1] - 1)))  # -1 for 'neck'
            # scores.append(person_score * max(0, pose_entries[n][-1]))
        return np.asarray(coco_keypoints), np.asarray(scores)
