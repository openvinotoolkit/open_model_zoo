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
from .base_profiler import MetricProfiler
from ...utils import contains_all


class DetectionProfiler(MetricProfiler):
    __provider__ = 'detection'

    def __init__(self, dump_iterations=100, report_type='csv'):
        self.names = []
        self.metric_names = []
        if report_type == 'csv':
            self.fields = ['identifier', 'label', 'score', 'pred', 'gt', 'matched']
        else:
            self.fields = ['identifier', 'per_class_result']
        self.updated_fields = False

        super().__init__(dump_iterations, report_type)

    def generate_profiling_data(self, identifier, metric_result, metric_name, final_score):
        if not self.updated_fields:
            self._update_fields()
        if self._last_profile and self._last_profile == identifier:
            report = self._last_profile
        else:
            report = self.per_box_result(identifier, metric_result) if self.report_file == 'csv' else {}

        if self.report_type == 'json':
            report = self.generate_json_report(identifier, metric_result, metric_name)
            report['{}_result'.format(metric_name)] = final_score
            return report

        if isinstance(report, list):
            for string in report:
                string['{}_result'.format(metric_name)] = metric_result[string['label']]['result']
        else:
            report['{}_result'.format(metric_name)] = metric_result['result']

        return report

    def generate_json_report(self, identifier, metric_result, metric_name):
        report = {'identifier': identifier, 'per_class_result': {}}
        per_class_results = {}
        for idx, (class_id, class_result) in enumerate(metric_result.items()):
            if not np.size(class_result['scores']):
                continue
            label_id = self.valid_labels[idx] if self.valid_labels else class_id
            iou = [iou_str.tolist() for iou_str in class_result['iou']]
            gt = class_result['gt'].tolist() if not isinstance(class_result['gt'], list) else class_result['gt']
            dt = class_result['dt'].tolist() if not isinstance(class_result['dt'], list) else class_result['dt']
            scores = (
                class_result['scores'].tolist()
                if not isinstance(class_result['scores'], list) else class_result['scores']
            )

            per_class_results[label_id] = {
                'annotation_boxes': gt,
                'prediction_boxes': dt,
                'prediction_scores': scores,
                'iou': iou,
            }
            per_class_results[label_id].update(self.generate_result_matching(class_result, metric_name))
        report['per_class_result'] = per_class_results
        return report

    def per_box_result(self, identifier, metric_result):
        per_box_results = []
        for label, per_class_result in enumerate(metric_result):
            if not np.size(per_class_result['scores']):
                continue
            label_id = self.valid_labels[label] if self.valid_labels else label
            scores = per_class_result['scores']
            dt = per_class_result['dt']
            gt = per_class_result['gt']
            matches_result = self.generate_result_matching(per_class_result, '')
            dt_matched = matches_result['dt_matches']
            gt_matched = matches_result['gt_matches']
            for dt_id, dt_box in enumerate(dt):
                box_result = {
                    'identifier': identifier,
                    'label': label_id,
                    'score': scores[dt_id],
                    'pred': dt_box,
                    'gt': ''
                }
                if dt_matched[dt_id]:
                    gt_id = np.where(gt_matched == dt_id + 1)
                    box_result['gt'] = gt[gt_id]
                per_box_results.append(box_result)
            for gt_id, gt_box in enumerate(gt):
                if gt_matched[gt_id] == -1:
                    box_result = {
                        'identifier': identifier,
                        'label': label_id,
                        'score': '',
                        'pred': '',
                        'gt': gt_box
                    }
                    per_box_results.append(box_result)

        return per_box_results

    def set_dataset_meta(self, meta):
        super().set_dataset_meta(meta)
        self.valid_labels = [
            label for label in meta.get('label_map', {}) if label != meta.get('background_label')
        ]

    @staticmethod
    def generate_result_matching(per_class_result, metric_name):
        if contains_all(['gt_matches', 'dt_matches'], per_class_result):
            matching_result = {
                'prediction_matches': per_class_result['dt_matches'][0],
                'annotation_matches':  per_class_result['gt_matches'][0],
                metric_name: per_class_result['result']
                }
            return matching_result
        matches = per_class_result['matched']
        dt_matches = np.zeros_like(per_class_result['scores'], dtype=int)
        gt_matches = [[] for _ in range(np.size(per_class_result['gt']))]
        for dt, value in matches.items():
            dt_matches[dt] = 1
            gt_list = value[0].tolist()
            for gt in gt_list:
                gt_matches[gt].append(dt)
        matching_result = {
            'prediction_matches': dt_matches.tolist(),
            'annotation_matches': gt_matches,
            'precision': per_class_result['precision'].tolist(),
            'recall': per_class_result['recall'].tolist(),
            'fppi': per_class_result['fppi'].tolist()
        }
        return matching_result

    def register_metric(self, metric_name):
        self.metric_names.append(metric_name)

    def _update_fields(self):
        for metric_name in self.metric_names:
            self.fields.append('{}_result'.format(metric_name))
        self.updated_fields = True
