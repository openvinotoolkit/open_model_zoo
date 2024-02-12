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
from .base_profiler import MetricProfiler
from ...utils import contains_all


class DetectionProfiler(MetricProfiler):
    __provider__ = 'detection_voc'

    def __init__(self, dump_iterations=100, report_type='csv', name=None):
        self.names = []
        self.metric_names = []
        if report_type == 'csv':
            self.fields = ['identifier', 'label', 'score', 'pred', 'gt', 'matched']
        else:
            self.fields = ['identifier', 'per_class_result']
        self.updated_fields = False

        super().__init__(dump_iterations, report_type, name)

    def generate_profiling_data(self, identifier, metric_result, metric_name, final_score):
        if not self.updated_fields:
            self._update_fields()
        if self._last_profile and self._last_profile == identifier:
            report = self._last_profile
        else:
            report = self.per_box_result(identifier, metric_result) if self.report_type == 'csv' else {}

        if self.report_type == 'json':
            report = self.generate_json_report(identifier, metric_result, metric_name)
            report['result'] = final_score if not np.isnan(final_score) else -1
            return report

        if isinstance(report, list):
            for string in report:
                res = metric_result[string['label']]['result']
                string['{}_result'.format(metric_name)] = res if not np.isnan(res) else -1
        else:
            report['{}_result'.format(metric_name)] = (
                metric_result['result'] if not np.isnan(metric_result['result']) else -1
            )

        return report

    def generate_json_report(self, identifier, metric_result, metric_name):
        report = {'identifier': identifier, 'per_class_result': {}}
        totat_pred_boxes, total_gt_boxes, total_gt_matches, total_pred_matches = 0, 0, 0, 0
        per_class_results = {}
        for idx, (class_id, class_result) in enumerate(metric_result.items()):
            if not np.size(class_result['gt']) + np.size(class_result['dt']):
                continue
            label_id = self.valid_labels[idx] if self.valid_labels and idx < len(self.valid_labels) else class_id
            iou = [iou_str.tolist() for iou_str in class_result['iou']]
            gt = class_result['gt'].tolist() if not isinstance(class_result['gt'], list) else class_result['gt']
            dt = class_result['dt'].tolist() if not isinstance(class_result['dt'], list) else class_result['dt']
            scores = (
                class_result['scores'].tolist()
                if not isinstance(class_result['scores'], list) else class_result['scores']
            )
            total_gt_boxes += len(gt)
            totat_pred_boxes += len(dt)

            per_class_results[label_id] = {
                'annotation_boxes': gt,
                'prediction_boxes': dt,
                'prediction_scores': scores,
                'iou': iou,
                'num_prediction_boxes': len(dt),
                'num_annotation_boxes': len(gt)
            }
            per_class_results[label_id].update(self.generate_result_matching(class_result, metric_name))
            total_pred_matches += per_class_results[label_id]['prediction_matches']
            total_gt_matches += per_class_results[label_id]['annotation_matches']
        report['per_class_result'] = per_class_results
        report['num_prediction_boxes'] = totat_pred_boxes
        report['num_annotation_boxes'] = total_gt_boxes
        report['total_annotation_matches'] = total_gt_matches
        report['total_prediction_matches'] = total_pred_matches
        return report

    def per_box_result(self, identifier, metric_result):
        per_box_results = []
        if isinstance(metric_result, dict):
            is_metric_result_dict = True
            unpacked_result = metric_result.items()
        else:
            is_metric_result_dict = False
            unpacked_result = enumerate(metric_result)
        for label, per_class_result in unpacked_result:
            if not np.size(per_class_result['scores']):
                continue
            label_id = self.valid_labels[label] if self.valid_labels and not is_metric_result_dict else label
            scores = per_class_result['scores']
            dt = per_class_result['dt']
            gt = per_class_result['gt']
            for dt_id, dt_box in enumerate(dt):
                box_result = {
                    'identifier': identifier,
                    'label': label_id,
                    'score': scores[dt_id],
                    'pred': dt_box,
                    'gt': ''
                }
                per_box_results.append(box_result)
            for gt_box in gt:
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
        if contains_all(per_class_result, ['gt_matches', 'dt_matches']):
            dt_matches = np.sum(per_class_result['dt_matches'][0] != 0)
            gt_matches = np.unique(np.argwhere(per_class_result['dt_matches'][0] != 0)).size
            matching_result = {
                'prediction_matches': int(dt_matches),
                'annotation_matches': gt_matches,
                'precision': per_class_result['precision'] if not np.isnan(per_class_result['precision']) else -1,
                'recall': per_class_result['recall'] if not np.isnan(per_class_result['recall']) else -1,

                metric_name: per_class_result['result'] if not np.isnan(per_class_result['result']) else -1
            }
            if 'ap' in per_class_result:
                matching_result['ap'] = per_class_result['ap'] if not np.isnan(per_class_result['ap']) else -1
            return matching_result
        matches = per_class_result['matched']
        dt_matches = 0
        gt_matches = 0
        for _, value in matches.items():
            dt_matches += 1
            gt_matches += len(value[0])

        precision = per_class_result['precision'][-1] if np.size(per_class_result['precision']) else -1
        recall = per_class_result['recall'][-1] if np.size(per_class_result['recall']) else -1
        matching_result = {
            'prediction_matches': dt_matches,
            'annotation_matches': gt_matches,
            'precision': precision if not np.isnan(precision) else -1,
            'recall': recall if not np.isnan(recall) else -1
        }
        if 'ap' in per_class_result:
            matching_result['ap'] = per_class_result['ap'] if not np.isnan(per_class_result['ap']) else -1
        return matching_result

    def register_metric(self, metric_name):
        self.metric_names.append(metric_name)

    def _update_fields(self):
        for metric_name in self.metric_names:
            self.fields.append('{}_result'.format(metric_name))
        self.updated_fields = True


class DetectionListProfiler(DetectionProfiler):
    __provider__ = 'detection_coco'

    def generate_json_report(self, identifier, metric_result, metric_name):
        report = {'identifier': identifier, 'per_class_result': {}}
        per_class_results = {}
        totat_pred_boxes, total_gt_boxes, total_gt_matches, total_pred_matches = 0, 0, 0, 0
        for idx, class_result in metric_result.items():
            if not np.size(class_result['gt']) + np.size(class_result['dt']):
                continue
            label_id = idx
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
                'num_prediction_boxes': len(dt),
                'num_annotation_boxes': len(gt),
                'prediction_scores': scores,
                'iou': iou,
            }
            total_gt_boxes += len(gt)
            totat_pred_boxes += len(dt)
            per_class_results[label_id].update(self.generate_result_matching(class_result, metric_name))
            total_pred_matches += per_class_results[label_id]['prediction_matches']
            total_gt_matches += per_class_results[label_id]['annotation_matches']
        report['per_class_result'] = per_class_results
        report['num_prediction_boxes'] = totat_pred_boxes
        report['num_annotation_boxes'] = total_gt_boxes
        report['total_annotation_matches'] = total_gt_matches
        report['total_prediction_matches'] = total_pred_matches
        return report
