"""
Copyright (c) 2018-2020 Intel Corporation

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
from .metric import PerImageEvaluationMetric
from ..representation import SalientRegionAnnotation, SalientRegionPrediction


class SalienceMapMAE(PerImageEvaluationMetric):
    __provider__ = 'salience_mae'
    annotation_types = (SalientRegionAnnotation, )
    prediction_types = (SalientRegionPrediction, )

    def configure(self):
        self.errors = []
        self.meta.update({
            'names': ['mean', 'std'],
            'scale': 1,
            'postfix': ' ',
            'calculate_mean': False,
            'target': 'higher-worse'
        })
        self.cnt = 0

    def update(self, annotation, prediction):
        self.cnt += 1
        if np.max(annotation.mask) == 0 or np.max(prediction.mask) == 0:
            return 0
        foreground_pixels = prediction.mask[np.where(annotation.mask)]
        foreground_error = np.size(foreground_pixels) - np.sum(foreground_pixels)
        background_error = np.sum(prediction.mask[np.where(~annotation.mask)])
        mae = (foreground_error + background_error) / np.size(annotation.mask)
        self.errors.append(mae)
        return mae

    def evaluate(self, annotations, predictions):
        return np.mean(self.errors), np.std(self.errors)

    def reset(self):
        del self.errors
        self.errors = []
        self.cnt = 0


class SalienceMapFMeasure(PerImageEvaluationMetric):
    __provider__ = 'salience_f-measure'
    annotation_types = (SalientRegionAnnotation, )
    prediction_types = (SalientRegionPrediction, )

    def configure(self):
        self.recalls, self.precisions, self.fmeasure = [], [], []
        self.meta.update({
            'names': ['recall', 'precision', 'f-measure'],
            'calculate_mean': False,
        })

    def update(self, annotation, prediction):
        sum_label = 2 * np.mean(prediction.mask)
        if sum_label > 1:
            sum_label = 1

        label3 = np.zeros_like(annotation.mask)
        label3[prediction.mask >= sum_label] = 1

        num_recall = np.sum(label3 == 1)
        label_and = np.logical_and(label3, annotation.mask)
        num_and = np.sum(label_and == 1)
        num_obj = np.sum(annotation.mask)

        if num_and == 0:
            self.recalls.append(0)
            self.precisions.append(0)
            self.fmeasure.append(0)
            return 0
        precision = num_and / num_recall if num_recall != 0 else 0
        recall = num_and / num_obj if num_obj != 0 else 0
        fmeasure = (1.3 * precision * recall) / (0.3 * precision + recall) if precision + recall != 0 else 0
        self.recalls.append(recall)
        self.precisions.append(precision)
        self.fmeasure.append(fmeasure)
        return fmeasure

    def evaluate(self, annotations, predictions):
        return np.mean(self.recalls), np.mean(self.precisions), np.mean(self.fmeasure)

    def reset(self):
        self.recalls, self.precisions, self.fmeasure = [], [], []


class SalienceEMeasure(PerImageEvaluationMetric):
    __provider__ = 'salience_e-measure'
    annotation_types = (SalientRegionAnnotation, )
    prediction_types = (SalientRegionPrediction, )

    def configure(self):
        self.scores = []

    def update(self, annotation, prediction):
        if np.sum(annotation.mask) == 0:
            enhance_matrix = 1 - prediction.mask
        elif np.sum(~annotation.mask) == 0:
            enhance_matrix = prediction.mask
        else:
            align_matrix = self.alignment_term(prediction.mask, annotation.mask)
            enhance_matrix = ((align_matrix + 1)**2) / 4
        h, w = annotation.mask.shape[:2]
        score = np.sum(enhance_matrix)/(w * h + np.finfo(float).eps)
        self.scores.append(score)
        return score

    @staticmethod
    def alignment_term(pred_mask, gt_mask):
        mu_fm = np.mean(pred_mask)
        mu_gt = np.mean(gt_mask)
        align_fm = pred_mask - mu_fm
        align_gt = gt_mask - mu_gt

        align_matrix = 2. * (align_gt * align_fm) / (align_gt * align_gt + align_fm * align_fm + np.finfo(float).eps)
        return align_matrix

    def evaluate(self, annotations, predictions):
        return np.mean(self.scores)

    def reset(self):
        self.scores = []
