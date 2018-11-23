"""
 Copyright (c) 2018 Intel Corporation

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
import pytest
from unittest.mock import MagicMock

from contextlib import contextmanager
from pathlib import Path, PosixPath

from tempfile import TemporaryDirectory
import numpy as np

from accuracy_checker.representation import DetectionAnnotation, DetectionPrediction


@contextmanager
def mock_filesystem(hierarchy):
    with TemporaryDirectory() as prefix:
        for entry in hierarchy:
            path = Path(prefix) / entry
            if entry.endswith("/"):
                path.mkdir(parents=True, exist_ok=True)
            else:
                parent = path.parent
                if parent != Path("."):
                    parent.mkdir(parents=True, exist_ok=True)
                # create file
                path.open('w').close()

        yield prefix


def make_representation(bounding_boxes, is_ground_truth=False, score=None):
    """
    Args:
        bounding_boxes: string or list of strings `score label x0 y0 x1 y1; label score x0 y0 x1 y1; ...`
        is_ground_truth: True if bbs are annotation boxes
        score: value in [0, 1], if not None, all prediction boxes are considered with the given score
    """
    if not isinstance(bounding_boxes, list):
        bounding_boxes = [bounding_boxes]
    res = []
    for i, bb in enumerate(bounding_boxes):
        arr = np.array(np.mat(bb))

        if bb == "":
            arr = np.array([]).reshape((0, 5))

        if is_ground_truth or score is not None:
            assert arr.shape[1] == 5
        elif not is_ground_truth and score is None:
            assert arr.shape[1] == 6
        if not is_ground_truth and score is not None:
            arr = np.c_[np.full(arr.shape[0], score), arr]

        if is_ground_truth:
            r = DetectionAnnotation(str(i), arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4])
        else:
            r = DetectionPrediction(str(i), arr[:, 1], arr[:, 0], arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5])
        res.append(r)
    return res


def update_dict(d, **kwargs):
    d = d.copy()
    d.update(**kwargs)
    return d
