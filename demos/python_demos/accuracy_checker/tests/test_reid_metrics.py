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
import numpy as np
from accuracy_checker.metrics import CMCScore
from accuracy_checker.metrics.reid import _eval_cmc


class TestCMC():
    def test_only_distmat(self):
        distmat = np.array([
            [0, 1, 2, 3, 4],
            [1, 0, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 0]
        ])
        m, n = distmat.shape
        query_ids = np.arange(m)
        gallery_ids = np.arange(n)
        query_cams = np.zeros(m).astype(np.int32)
        gallery_cams = np.ones(n).astype(np.int32)
        ret = _eval_cmc(distmat, query_ids=query_ids, gallery_ids=gallery_ids, query_cams=query_cams, gallery_cams=gallery_cams)
        assert np.all(ret[:5] == [0.6, 0.6, 0.8, 1.0, 1.0])

    def test_duplicate_ids(self):
        distmat = np.array([
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3]
        ])
        query_ids = np.array([0, 0, 1, 1])
        gallery_ids = np.array([0, 0, 1, 1])
        gallery_cams = np.ones(distmat.shape[1]).astype(np.int32)
        query_cams = np.zeros(distmat.shape[0]).astype(np.int32)
        ret = _eval_cmc(
            distmat, query_ids=query_ids, gallery_ids=gallery_ids, topk=4, gallery_cams=gallery_cams, query_cams=query_cams,
            separate_camera_set=False, single_gallery_shot=False
        )
        assert np.all(ret == [0.5, 0.5, 1, 1])

    def test_duplicate_cams(self):
        distmat = np.tile(np.arange(5), (5, 1))
        query_ids = np.array([0,0,0,1,1])
        gallery_ids = np.array([0,0,0,1,1])
        query_cams = np.array([0,0,0,0,0])
        gallery_cams = np.array([0,1,1,1,1])
        ret = _eval_cmc(
            distmat, query_ids=query_ids, gallery_ids=gallery_ids,
            query_cams=query_cams, gallery_cams=gallery_cams, topk=5,
            separate_camera_set=False, single_gallery_shot=False
        )
        assert np.all(ret == [0.6, 0.6, 0.6, 1, 1])
