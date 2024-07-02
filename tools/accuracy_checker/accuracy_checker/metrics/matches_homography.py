"""
Copyright (c) 2024 Intel Corporation

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
from collections import defaultdict
import collections.abc as collections
import numpy as np
from .metric import Metric
from ..representation import ImageFeatureAnnotation, ImageFeaturePrediction
from ..utils import UnsupportedPackage
try:
    import torch
except ImportError as err_torch:
    torch = UnsupportedPackage("torch", err_torch)


string_classes = (str, bytes)

def map_tensor(input_, func):
    if isinstance(input_, string_classes):
        return input_
    if isinstance(input_, collections.Mapping):
        return {k: map_tensor(sample, func) for k, sample in input_.items()}
    if isinstance(input_, collections.Sequence):
        return [map_tensor(sample, func) for sample in input_]
    if input_ is None:
        return None
    return func(input_)


def index_batch(tensor_dict):
    batch_size = len(next(iter(tensor_dict.values()), None))
    for i in range(batch_size):
        yield map_tensor(tensor_dict, lambda t, idx=i: t[idx])


def to_homogeneous(points):
    """Convert N-dimensional points to homogeneous coordinates.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N).
    Returns:
        A torch.Tensor or numpy.ndarray with size (..., N+1).
    """
    if isinstance(points, torch.Tensor):
        pad = points.new_ones(points.shape[:-1] + (1,))
        return torch.cat([points, pad], dim=-1)
    if isinstance(points, np.ndarray):
        pad = np.ones((points.shape[:-1] + (1,)), dtype=points.dtype)
        return np.concatenate([points, pad], axis=-1)
    raise ValueError


def from_homogeneous(points, eps=0.0):
    """Remove the homogeneous dimension of N-dimensional points.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N+1).
        eps: Epsilon value to prevent zero division.
    Returns:
        A torch.Tensor or numpy ndarray with size (..., N).
    """
    return points[..., :-1] / (points[..., -1:] + eps)


def sym_homography_error(kpts0, kpts1, T_0to1):
    kpts0_1 = from_homogeneous(to_homogeneous(kpts0) @ T_0to1.transpose(-1, -2))
    dist0_1 = np.sqrt(((kpts0_1 - kpts1) ** 2).sum(-1))

    kpts1_0 = from_homogeneous(
        to_homogeneous(kpts1) @ np.linalg.pinv(T_0to1.transpose(-1, -2))
    )
    dist1_0 = np.sqrt(((kpts1_0 - kpts0) ** 2).sum(-1))

    return ((dist0_1 + dist1_0) / 2.0).detach().cpu().numpy()


def check_keys_recursive(d, pattern):
    if isinstance(pattern, dict):
        for k, v in pattern.items():
            check_keys_recursive(d[k], v)
    else:
        for k in pattern:
            assert k in d.keys()

def get_matches_scores(kpts0, kpts1, matches0, mscores0):
    m0 = matches0 > -1
    m1 = matches0[m0]
    pts0 = kpts0[m0]
    pts1 = kpts1[m1]
    scores = mscores0[m0]
    return pts0, pts1, scores


def eval_per_batch_item(data: dict, pred: dict, eval_f, *args, **kwargs):
    # Batched data
    results = [
        eval_f(data_i, pred_i, *args, **kwargs)
        for data_i, pred_i in zip(index_batch(data), index_batch(pred))
    ]
    # Return a dictionary of lists with the evaluation of each item
    return {k: [r[k] for r in results] for k in results[0].keys()}


# See https://github.com/cvg/glue-factory/blob/main/gluefactory/eval/utils.py

def eval_matches_homography(kp0, kp1, m0, scores0, H_gt) -> dict:
    pts0, pts1, _ = get_matches_scores(kp0, kp1, m0, scores0)
    err = sym_homography_error(pts0, pts1, H_gt)
    results = {}

    results["prec@1px"] = np.nan_to_num((err < 1).astype(float).mean()).item()
    results["prec@3px"] = np.nan_to_num((err < 3).astype(float).mean()).item()
    results["num_matches"] = pts0.shape[0]
    results["num_keypoints"] = (kp0.shape[0] + kp1.shape[0]) / 2.0
    return results

class MatchesHomography(Metric):
    __provider__ = 'matches_homography'
    annotation_types = (ImageFeatureAnnotation)
    prediction_types = (ImageFeaturePrediction)

    def configure(self):
        self.metrics = defaultdict(list)

    def update(self, annotation, prediction):
        annotations = annotation.identifier.data_id

        H_gt = annotations["H_0to1"]
        kp0, kp1 = annotations["keypoints0"][0], annotations["keypoints1"][0]
        m0, scores0 = prediction["matches0"][0], prediction["matching_scores0"][0]

        results = eval_matches_homography(kp0, kp1, m0, scores0, H_gt)

        results["name"] = annotation.identifier

        for k, v in results.items():
            self.metrics[k].append(v)

        return [results["prec@1px"], results["prec@3px"], results["num_matches"] / results["num_keypoints"]]

    def evaluate(self, annotations, predictions):
        summary = {}
        for k, v in self.metrics.items():
            arr = np.array(v)
            if not np.issubdtype(np.array(v).dtype, np.number):
                continue
            summary[k] = round(np.median(arr), 3)

        return summary["prec@1px"], summary["prec@3px"], summary["num_matches"]/summary["num_keypoints"]

    def reset(self):
        self.metrics = []

    @classmethod
    def get_common_meta(cls):
        return {
            'target': 'higher-better',
            'names': ['prec@1px', 'prec@3px', 'num_matches'],
        }
