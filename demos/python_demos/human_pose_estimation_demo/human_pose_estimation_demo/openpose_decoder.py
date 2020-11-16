import numpy as np


class OpenPoseDecoder:

    BODY_PARTS_KPT_IDS = ((1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
                          (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17))
    BODY_PARTS_PAF_IDS = (12, 20, 14, 16, 22, 24, 0, 2, 4, 6, 8, 10, 28, 30, 34, 32, 36, 18, 26)

    def __init__(self, num_joints=18, skeleton=BODY_PARTS_KPT_IDS, paf_indices=BODY_PARTS_PAF_IDS,
                 max_points=100, score_threshold=0.1, min_paf_alignment_score=0.05, delta=0.5):
        self.num_joints = num_joints
        self.skeleton = skeleton
        self.paf_indices = paf_indices
        self.max_points = max_points
        self.score_threshold = score_threshold
        self.min_paf_alignment_score = min_paf_alignment_score
        self.delta = delta

        self.points_per_limb = 10
        self.grid = np.arange(self.points_per_limb, dtype=np.float32).reshape(1, -1, 1)

    def __call__(self, heatmaps, nms_heatmaps, pafs):
        batch_size, _, h, w = heatmaps.shape
        assert batch_size == 1, 'Batch size of 1 only supported'

        keypoints = self.extract_points(heatmaps, nms_heatmaps)
        pafs = np.transpose(pafs, (0, 2, 3, 1))

        if self.delta > 0:
            for kpts in keypoints:
                kpts[:, :2] += self.delta
                np.core.umath.clip(kpts[:, 0], 0, w - 1, out=kpts[:, 0])
                np.core.umath.clip(kpts[:, 1], 0, h - 1, out=kpts[:, 1])

        pose_entries, keypoints = self.group_keypoints(keypoints, pafs, pose_entry_size=self.num_joints + 2)
        poses, scores = self.convert_to_coco_format(pose_entries, keypoints)
        if len(poses) > 0:
            poses = np.asarray(poses, dtype=np.float32)
            poses = poses.reshape((poses.shape[0], -1, 3))
        else:
            poses = np.empty((0, 17, 3), dtype=np.float32)
            scores = np.empty(0, dtype=np.float32)

        return poses, scores

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
                all_keypoints.append(np.empty((0, 4), dtype=np.float32))
                continue
            # Apply quarter offset to improve localization accuracy.
            x, y = self.refine(heatmaps[0, k], x, y)
            np.core.umath.clip(x, 0, w - 1, out=x)
            np.core.umath.clip(y, 0, h - 1, out=y)
            # Pack resulting points.
            keypoints = np.empty((n, 4), dtype=np.float32)
            keypoints[:, 0] = x
            keypoints[:, 1] = y
            keypoints[:, 2] = score
            keypoints[:, 3] = np.arange(keypoint_id, keypoint_id + n)
            keypoint_id += n
            all_keypoints.append(keypoints)
        return all_keypoints

    def top_k(self, heatmaps):
        N, K, _, W = heatmaps.shape
        heatmaps = heatmaps.reshape(N, K, -1)
        # Get positions with top scores.
        ind = heatmaps.argpartition(-self.max_points, axis=2)[:, :, -self.max_points:]
        scores = np.take_along_axis(heatmaps, ind, axis=2)
        # Keep top scores sorted.
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

    def update_poses(self, part_id, kpt_a_id, kpt_b_id, all_keypoints, connections, pose_entries, pose_entry_size):
        if part_id == 0:
            pose_entries = [np.full(pose_entry_size, -1, dtype=np.float32) for _ in range(len(connections))]
            for pose, connection in zip(pose_entries, connections):
                pose[kpt_a_id] = connection[0]
                pose[kpt_b_id] = connection[1]
                pose[-1] = 2
                # pose score = sum of all points' scores + sum of all connections' scores
                pose[-2] = np.sum(all_keypoints[connection[0:2], 2]) + connection[2]
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
                    # Create new pose entry.
                    pose_entry = np.full(pose_entry_size, -1)
                    pose_entry[kpt_a_id] = connection[0]
                    pose_entry[kpt_b_id] = connection[1]
                    pose_entry[-1] = 2
                    pose_entry[-2] = np.sum(all_keypoints[connection[0:2], 2]) + connection[2]
                    pose_entries.append(pose_entry)
                elif pose_a_idx >= 0 and pose_b_idx >= 0:
                    if pose_a_idx != pose_b_idx:
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
                        pose_a[-1] += pose_b[-1]
                        pose_a[-2] += pose_b[-2] + connection[2]
                        del pose_entries[pose_b_idx]
                    else:
                        # Adjust score of a pose.
                        pose_entries[pose_a_idx][-2] += connection[2]
                elif pose_a_idx >= 0:
                    # Add a new limb into pose.
                    pose = pose_entries[pose_a_idx]
                    if pose[kpt_b_id] < 0:
                        pose[-2] += all_keypoints[connection[1], 2]
                    pose[kpt_b_id] = connection[1]
                    pose[-2] += connection[2]
                    pose[-1] += 1
                elif pose_b_idx >= 0:
                    # Add a new limb into pose.
                    pose = pose_entries[pose_b_idx]
                    if pose[kpt_a_id] < 0:
                        pose[-2] += all_keypoints[connection[0], 2]
                    pose[kpt_a_id] = connection[0]
                    pose[-2] += connection[2]
                    pose[-1] += 1
        return pose_entries

    def group_keypoints(self, all_keypoints_by_type, pafs, pose_entry_size=20):
        all_keypoints = np.concatenate(all_keypoints_by_type, axis=0)
        pose_entries = []
        # For every limb.
        for part_id, paf_channel in enumerate(self.paf_indices):
            kpt_a_id, kpt_b_id = self.skeleton[part_id]
            kpts_a = all_keypoints_by_type[kpt_a_id]
            kpts_b = all_keypoints_by_type[kpt_b_id]
            n = len(kpts_a)
            m = len(kpts_b)
            if n == 0 or m == 0:
                continue

            # Get vectors between all pairs of keypoints, i.e. candidate limb vectors.
            a = kpts_a[:, :2]
            a = np.broadcast_to(a[None], (m, n, 2))
            b = kpts_b[:, :2]
            vec_raw = (b[:, None, :] - a).reshape(-1, 1, 2)

            # Sample points along every candidate limb vector.
            steps = (1 / (self.points_per_limb - 1) * vec_raw)
            points = steps * self.grid + a.reshape(-1, 1, 2)
            points = points.round().astype(dtype=np.int32)
            x = points[..., 0].ravel()
            y = points[..., 1].ravel()

            # Compute affinity score between candidate limb vectors and part affinity field.
            part_pafs = pafs[0, :, :, paf_channel:paf_channel + 2]
            field = part_pafs[y, x].reshape(-1, self.points_per_limb, 2)
            vec_norm = np.linalg.norm(vec_raw, ord=2, axis=-1, keepdims=True)
            vec = vec_raw / (vec_norm + 1e-6)
            affinity_scores = (field * vec).sum(-1).reshape(-1, self.points_per_limb)
            valid_affinity_scores = affinity_scores > self.min_paf_alignment_score
            valid_num = valid_affinity_scores.sum(1)
            affinity_scores = (affinity_scores * valid_affinity_scores).sum(1) / (valid_num + 1e-6)
            success_ratio = valid_num / self.points_per_limb

            # Get a list of limbs according to the obtained affinity score.
            valid_limbs = np.where(np.logical_and(affinity_scores > 0, success_ratio > 0.8))[0]
            affinity_scores = affinity_scores[valid_limbs]
            b_idx, a_idx = np.divmod(valid_limbs, n)
            if len(affinity_scores) == 0:
                continue

            # From all retrieved connections that share starting/ending keypoints leave only the top-scoring ones.
            order = affinity_scores.argsort()[::-1]
            affinity_scores = affinity_scores[order]
            a_idx = a_idx[order]
            b_idx = b_idx[order]
            a_idx_unique = np.unique(a_idx, return_index=True)[1]
            b_idx_unique = np.unique(b_idx, return_index=True)[1]
            idx = np.intersect1d(a_idx_unique, b_idx_unique, assume_unique=True)
            a = kpts_a[a_idx[idx], 3].astype(np.int32)
            b = kpts_b[b_idx[idx], 3].astype(np.int32)
            connections = list(zip(a, b, affinity_scores[idx]))

            if len(connections) == 0:
                continue

            # Update poses with new connections.
            pose_entries = self.update_poses(part_id, kpt_a_id, kpt_b_id, all_keypoints,
                                             connections, pose_entries, pose_entry_size)

        # Remove poses with not enough points.
        filtered_entries = []
        for i in range(len(pose_entries)):
            if pose_entries[i][-1] < 3:
                continue
            filtered_entries.append(pose_entries[i])
        pose_entries = np.asarray(filtered_entries, dtype=np.float32)
        return pose_entries, all_keypoints

    def convert_to_coco_format(self, pose_entries, all_keypoints):
        num_joints = 17
        coco_keypoints = []
        scores = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            keypoints = np.zeros(num_joints * 3)
            reorder_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
            person_score = pose_entries[n][-2]
            for keypoint_id, target_id in zip(pose_entries[n][:-2], reorder_map):
                if target_id < 0:
                    continue
                cx, cy, score = 0, 0, 0  # keypoint not found
                if keypoint_id != -1:
                    cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                keypoints[target_id * 3 + 0] = cx
                keypoints[target_id * 3 + 1] = cy
                keypoints[target_id * 3 + 2] = score
            coco_keypoints.append(keypoints)
            scores.append(person_score * max(0, (pose_entries[n][-1] - 1)))  # -1 for 'neck'
        return np.asarray(coco_keypoints), np.asarray(scores)
