#!/usr/bin/env python
import os
import numpy as np

import cv2 as cv
import cv2.open_model_zoo as omz

from tests_common import NewOpenCVTests, unittest


def rotatedRectIOU(a, b):
    res, inter = cv.rotatedRectangleIntersection(a, b)
    if inter is None or res == cv.INTERSECT_NONE:
        return 0.0
    if res == cv.INTERSECT_FULL:
        return 1.0
    interArea = cv.contourArea(inter)
    return interArea / (a[2]*a[3] + b[2]*b[3] - interArea)


class omz_text_recognition_test(NewOpenCVTests):

    def setUp(self):
        super(omz_text_recognition_test, self).setUp()


    def test_TextRecognitionPipeline(self):
        img = cv.imread(self.find_file('cv/cloning/Mixed_Cloning/source1.png'))

        p = omz.TextRecognitionPipeline()
        p.setPixelLinkThresh(0.5)
        p.setPixelClassificationThresh(0.5)

        rects, texts, confs = p.process(img)

        refTexts = ["c57410", "jie", "howard"]
        refRects = [
              ((110.39253234863281, 45.5788459777832), (48.49958419799805, 153.86648559570312), -87.61405944824219),
              ((93.0, 102.5), (80.0, 43.0), -0.0),
              ((111.65045928955078, 152.82647705078125), (48.7945442199707, 173.10397338867188), -88.87670135498047)
        ]

        self.assertEqual(len(rects), len(texts))
        self.assertEqual(len(texts), len(refTexts))

        for refRect, refText in zip(refRects, refTexts):
            matched = False

            for rect, text in zip(rects, texts):
                if text == refText:
                    matched = True
                    self.assertGreater(rotatedRectIOU(rect, refRect), 0.99)
                    break
            self.assertTrue(matched, refText)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
