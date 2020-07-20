import numpy as np
from .base_profiler import MetricProfiler


class DetectionProfiler(MetricProfiler):
    __provider__ = 'detection'
    fields = ['identifier', 'label', 'score', 'pred', 'gt', 'matched']

    def generate_profiling_data(self, identifier, metric_result, metric_name, final_score):
        # if self._last_profile and self._last_profile == identifier:
        #     report = self._last_profile
        # else:
        if self.report_file == 'csv':
            report = self.per_box_result(identifier, metric_result)
        else:
            report = self.generate_json_report(identifier, metric_result)
            report['{}_result'.format(metric_name)] = final_score
            return report

        if isinstance(report, list):
            for string in report:
                string['{}_result'.format(metric_name)] = metric_result[string['label']]['result']
        else:
            report['{}_result'.format(metric_name)] = metric_result['result']

        return report

    def generate_json_report(self, identifier, metric_result):
        report = {'identifier': identifier, 'per_class_result': {}}
        per_class_results = {}
        for idx, class_result in enumerate(metric_result):
            if not np.size(class_result['scores']):
                continue
            label_id = self.valid_labels[idx] if self.valid_labels else idx
            iou = [iou_str.tolist() for iou_str in class_result['iou']]
            per_class_results[label_id] = {
                'annotation_boxes': class_result['gt'].tolist(),
                'prediction_boxes': class_result['dt'].tolist(),
                'prediction_scores': class_result['scores'].tolist(),
                'iou': iou,
                'prediction_matches': class_result['dt_matches'][0].tolist(),
                'annotation_matches': class_result['gt_matches'][0].tolist(),
                'average_precision': class_result['result']
            }
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
            dt_matched = per_class_result['dt_matches'][0]
            gt_matched = per_class_result['gt_matches'][0]
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
