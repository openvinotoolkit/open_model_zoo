#!/usr/bin/env python
import os
import numpy as np

import cv2 as cv
import cv2.open_model_zoo as omz

from tests_common import NewOpenCVTests, unittest

class omz_human_pose_estimation_test(NewOpenCVTests):

    def setUp(self):
        super(omz_human_pose_estimation_test, self).setUp()


    def test_TextRecognitionPipeline(self):
        img = cv.imread(self.find_file('gpu/lbpcascade/er.png'))

        p = omz.HumanPoseEstimation()
        poses = p.process(img)
        self.assertEqual(len(poses), 6)

        p.render(img, poses)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
