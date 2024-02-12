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
from .object_detection_metric_profiler import DetectionListProfiler


class InstanceSegmentationProfiler(DetectionListProfiler):
    __provider__ = 'instance_segmentation'

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
            report = self.per_instance_result(identifier, metric_result) if self.report_type == 'csv' else {}

        if self.report_type == 'json':
            report = self.generate_json_report(identifier, metric_result, metric_name)
            report['result'] = final_score
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
        totat_pred_boxes, total_gt_boxes, total_gt_matches, total_pred_matches = 0, 0, 0, 0
        for idx, class_result in metric_result.items():
            if not np.size(class_result['gt']) + np.size(class_result['dt']):
                continue
            label_id = idx
            iou = [iou_str.tolist() for iou_str in class_result['iou']]
            gt = [g.tolist() if not isinstance(g, list) else g for g in class_result['gt']]
            dt = []
            for dp in class_result['dt']:
                dt.append([d.tolist() if not isinstance(d, list) else d for d in dp])
            scores = (
                class_result['scores'].tolist()
                if not isinstance(class_result['scores'], list) else class_result['scores']
            )

            per_class_results[label_id] = {
                'annotation_polygons': gt,
                'prediction_polygons': dt,
                'num_prediction_polygons': len(dt),
                'num_annotation_polygons': len(gt),
                'prediction_scores': scores,
                'iou': iou,
            }
            total_gt_boxes += len(gt)
            totat_pred_boxes += len(dt)
            per_class_results[label_id].update(self.generate_result_matching(class_result, metric_name, aggregate=True))
            total_pred_matches += per_class_results[label_id]['prediction_matches']
            total_gt_matches += per_class_results[label_id]['annotation_matches']
        report['per_class_result'] = per_class_results
        report['num_prediction_polygons'] = totat_pred_boxes
        report['num_annotation_polygons'] = total_gt_boxes
        report['total_annotation_matches'] = total_gt_matches
        report['total_prediction_matches'] = total_pred_matches
        return report

    def per_instance_result(self, identifier, metric_result):
        per_box_results = []
        for label, per_class_result in metric_result.items():
            if not np.size(per_class_result['scores']):
                continue
            scores = per_class_result['scores']
            dt = per_class_result['dt']
            gt = per_class_result['gt']
            matches_result = self.generate_result_matching(per_class_result, '')
            dt_matched = matches_result.get('prediction_matches', [])
            gt_matched = matches_result.get('annotation_matches', [])
            for dt_id, dt_box in enumerate(dt):
                box_result = {
                    'identifier': identifier,
                    'label': label,
                    'score': scores[dt_id],
                    'pred': dt_box,
                    'gt': ''
                }
                if dt_matched[dt_id]:
                    gt_id = np.where(gt_matched == dt_id + 1)
                    box_result['gt'] = np.array(gt)[gt_id]
                per_box_results.append(box_result)
            for gt_id, gt_box in enumerate(gt):
                if gt_matched[gt_id] == -1:
                    box_result = {
                        'identifier': identifier,
                        'label': label,
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
    def generate_result_matching(per_class_result, metric_name, aggregate=False):
        result = per_class_result['result']
        if np.isnan(result):
            result = -1
        dt_matches = per_class_result['dt_matches'][0].tolist() if 'dt_matches' in per_class_result else []
        gt_matches = per_class_result['gt_matches'][0].tolist() if 'gt_matches' in per_class_result else []
        if aggregate:
            dt_matches = int(sum(dt_matches))
            gt_matches = int(np.sum(np.array(gt_matches) != 0))
        matching_result = {
            'prediction_matches': dt_matches,
            'annotation_matches': gt_matches,
            'result': result,
        }
        return matching_result

    def register_metric(self, metric_name):
        self.metric_names.append(metric_name)

    def _update_fields(self):
        for metric_name in self.metric_names:
            self.fields.append('{}_result'.format(metric_name))
        self.updated_fields = True
