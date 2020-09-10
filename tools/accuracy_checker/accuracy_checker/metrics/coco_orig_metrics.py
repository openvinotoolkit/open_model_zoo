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

import os
import tempfile
import json
from copy import deepcopy
import warnings
from pathlib import Path
import numpy as np

from ..representation import (
    DetectionPrediction,
    DetectionAnnotation,
    CoCoInstanceSegmentationAnnotation,
    CoCocInstanceSegmentationPrediction,
    PoseEstimationAnnotation,
    PoseEstimationPrediction
)
from ..logging import print_info
from ..config import BaseField, BoolField, ConfigError
from ..utils import get_or_parse_value, UnsupportedPackage
from .metric import FullDatasetEvaluationMetric, PerImageEvaluationMetric
from .coco_metrics import COCO_THRESHOLDS, process_threshold, compute_precision_recall

try:
    from pycocotools.coco import COCO
except ImportError as import_error:
    COCO = UnsupportedPackage("pycocotools.coco", import_error.msg)
try:
    from pycocotools.cocoeval import COCOeval as coco_eval
except ImportError as import_error:
    coco_eval = UnsupportedPackage("pycocotools.cocoeval", import_error.msg)
try:
    from pycocotools.mask import iou as iou_calc
except ImportError as import_error:
    iou_calc = UnsupportedPackage("pycocotools.mask", import_error.msg)

SHOULD_SHOW_PREDICTIONS = False
SHOULD_DISPLAY_DEBUG_IMAGES = False
if SHOULD_DISPLAY_DEBUG_IMAGES:
    import cv2


class MSCOCOorigBaseMetric(FullDatasetEvaluationMetric):
    annotation_types = (DetectionAnnotation, )
    prediction_types = (DetectionPrediction, )

    iou_type = 'bbox'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'threshold': BaseField(optional=True, default='.50:.05:.95', description='threshold for metric calculation')
        })

        parameters.update({
            'include_boundaries': BoolField(
                optional=True, default=True,
                description="Whether to add '1' when computing height and width from bounding box coordinates."),
        })

        return parameters

    def configure(self):
        threshold = process_threshold(self.get_value_from_config('threshold'))
        self.threshold = get_or_parse_value(threshold, COCO_THRESHOLDS)
        self.box_side_delta = int(self.get_value_from_config('include_boundaries'))
        if not self.dataset.metadata:
            raise ConfigError('coco orig metrics require dataset_meta'
                              'Please provide dataset meta file or regenerate annotation')

        if not self.dataset.metadata.get('label_map'):
            raise ConfigError('coco_orig metrics require label_map providing in dataset_meta'
                              'Please provide dataset meta file or regenerated annotation')
        if isinstance(COCO, UnsupportedPackage):
            COCO.raise_error("MSCOCOorigBaseMetric")
        label_map = self.dataset.metadata.get('label_map', {})
        self.labels = [
            label for label in label_map
            if label != self.dataset.metadata.get('background_label')
        ]

    @staticmethod
    def _iou_type_data_to_coco(data_to_store, data, box_side_delta):
        x_mins = data.x_mins.tolist()
        y_mins = data.y_mins.tolist()
        x_maxs = data.x_maxs.tolist()
        y_maxs = data.y_maxs.tolist()

        for data_record, x_min, y_min, x_max, y_max in zip(
                data_to_store, x_mins, y_mins, x_maxs, y_maxs
        ):
            width = x_max - x_min + box_side_delta
            height = y_max - y_min + box_side_delta
            area = width * height
            data_record.update({
                'bbox': [x_min, y_min, width, height],
                'area': area
            })

        return data_to_store

    def _prepare_generated_coco_structures(self, annotations):
        label_map = self.dataset.metadata.get('label_map')
        if self.dataset.metadata.get('background_label') is not None:
            label_map.pop(self.dataset.metadata.get('background_label'))

        coco_data_to_store = self._prepare_data_for_annotation_file(
            annotations, label_map)

        coco_annotation = self._create_json(coco_data_to_store)
        coco = COCO(str(coco_annotation))

        return coco

    @staticmethod
    def generate_map_pred_label_id_to_coco_cat_id(has_background, use_full_label_map):
        shift = 0 if has_background else 1
        max_cat = 90 if use_full_label_map else 80
        max_key = max_cat - shift
        res_map = {i: i + shift for i in range(0, max_key+1)}
        assert max(res_map.values()) == max_cat
        return res_map

    def _prepare_original_coco_structures(self):
        annotation_conversion_parameters = self.dataset.config.get('annotation_conversion')
        annotation_file = annotation_conversion_parameters.get('annotation_file')
        has_background = annotation_conversion_parameters.get('has_background', False)
        use_full_label_map = annotation_conversion_parameters.get('use_full_label_map', False)

        meta = self.dataset.metadata

        coco = COCO(str(annotation_file))
        assert 0 not in coco.cats.keys()
        coco_cat_name_to_id = {v['name']: k for k, v in coco.cats.items()}
        if has_background:
            assert 'background_label' in meta
            bg_lbl = meta['background_label']
            bg_name = meta['label_map'][bg_lbl]
            assert bg_name not in coco_cat_name_to_id
            coco_cat_name_to_id[bg_name] = bg_lbl
        else:
            assert 'background_label' not in meta

        if not use_full_label_map:
            map_pred_label_id_to_coco_cat_id = {k: coco_cat_name_to_id[v] for k, v in meta['label_map'].items()}
        else:
            map_pred_label_id_to_coco_cat_id = self.generate_map_pred_label_id_to_coco_cat_id(has_background,
                                                                                              use_full_label_map)
            for k, v in meta['label_map'].items():
                assert map_pred_label_id_to_coco_cat_id[k] == coco_cat_name_to_id[v], (
                    "k = {}, v = {}, map_pred_label_id_to_coco_cat_id[k] = {}, coco_cat_name_to_id[v] = {}".format(
                        k, v, map_pred_label_id_to_coco_cat_id[k], coco_cat_name_to_id[v]))

            assert all(map_pred_label_id_to_coco_cat_id[k] == coco_cat_name_to_id[v]
                       for k, v in meta['label_map'].items())

        map_coco_img_file_name_to_img_id = {os.path.basename(v['file_name']): v['id'] for v in coco.dataset['images']}
        assert len(map_coco_img_file_name_to_img_id) == len(coco.dataset['images']), "Image name duplications"

        return coco, map_coco_img_file_name_to_img_id, map_pred_label_id_to_coco_cat_id

    def _convert_data_to_generated_coco_format(self, predictions):
        coco_data_to_store = []
        for pred in predictions:
            prediction_data_to_store = []
            cur_name = Path(pred.identifier).name
            cur_img_id = (cur_name.split(".jpg")[0]).split("_")[-1]

            labels = pred.labels.tolist()
            scores = pred.scores.tolist()
            cur_num = len(labels)
            assert len(scores) == cur_num

            for (s, cur_cat) in zip(scores, labels):
                prediction_data_to_store.append({
                    'image_id': cur_img_id,
                    'score': s,
                    'category_id': cur_cat,
                    '_image_name_from_dataset': cur_name,
                })
            prediction_data_to_store = self._iou_type_data_to_coco(prediction_data_to_store, pred,
                                                                   self.box_side_delta)
            coco_data_to_store.extend(prediction_data_to_store)

        return coco_data_to_store

    def _convert_data_to_original_coco_format(self, predictions,
                                              map_coco_img_file_name_to_img_id, map_pred_label_id_to_coco_cat_id):
        coco_data_to_store = []
        for pred in predictions:
            prediction_data_to_store = []
            cur_name = Path(pred.identifier).name
            assert cur_name in map_coco_img_file_name_to_img_id
            cur_img_id = map_coco_img_file_name_to_img_id[cur_name]
            if pred.size == 0:
                continue

            labels = pred.labels.tolist()
            scores = pred.scores.tolist()
            cur_num = len(labels)
            assert len(scores) == cur_num

            coco_cats = [map_pred_label_id_to_coco_cat_id[lbl] for lbl in labels]

            for (s, cur_cat) in zip(scores, coco_cats):
                prediction_data_to_store.append({
                    'image_id': cur_img_id,
                    'score': s,
                    'category_id': cur_cat,
                    '_image_name_from_dataset': cur_name,
                })
            prediction_data_to_store = self._iou_type_data_to_coco(prediction_data_to_store, pred,
                                                                   self.box_side_delta)
            coco_data_to_store.extend(prediction_data_to_store)

        return coco_data_to_store

    def _iou_type_specific_coco_annotation(self, annotation_data_to_store, annotation):
        annotation_data_to_store = self._iou_type_data_to_coco(annotation_data_to_store, annotation,
                                                               self.box_side_delta)
        return annotation_data_to_store

    def _prepare_data_for_annotation_file(
            self, annotations, dataset_label_map
    ):
        coco_annotation_to_store = []
        coco_category_to_store = []
        coco_image_to_store = []
        coco_images = {}
        count = 1
        for annotation in annotations:
            annotation_data_to_store = []
            cur_name = Path(annotation.identifier).name
            cur_img_id = (cur_name.split(".jpg")[0]).split("_")[-1]

            labels = annotation.labels
            iscrowds = annotation.metadata.get('iscrowd', [0]*annotation.size)

            for cur_cat, iscrowd in zip(labels, iscrowds):
                annotation_data_to_store.append({
                    'id': count,
                    'image_id': cur_img_id,
                    'category_id': int(cur_cat),
                    '_image_name_from_dataset': cur_name,
                    'iscrowd': iscrowd
                })
                count += 1

            annotation_data_to_store = self._iou_type_specific_coco_annotation(annotation_data_to_store, annotation)
            coco_annotation_to_store.extend(annotation_data_to_store)

            coco_images[cur_img_id] = {
                'id': cur_img_id,
                'file_name': cur_name,
                'width': annotation.metadata.get('image_size')[0][0],
                'height': annotation.metadata.get('image_size')[0][1]
            }

        for cat, cat_name in dataset_label_map.items():
            coco_category_to_store.append({
                'id': cat,
                'name': cat_name
            })
        for value in coco_images.values():
            coco_image_to_store.append(value)

        coco_data_to_store = {
            'info': {},
            'images': coco_image_to_store,
            'annotations': coco_annotation_to_store,
            'licenses': [],
            'categories': coco_category_to_store
        }
        return coco_data_to_store

    @staticmethod
    def _create_json(coco_data_to_store):
        with tempfile.NamedTemporaryFile() as ftmp:
            json_file_to_store = ftmp.name + ".json"
        with open(json_file_to_store, 'w') as f:
            json.dump(coco_data_to_store, f, indent=4)

        return json_file_to_store

    def _reload_results_to_coco_class(self, coco, coco_data_to_store):
        json_file_to_load = self._create_json(coco_data_to_store)
        coco_res = coco.loadRes(json_file_to_load)

        return coco_res

    @staticmethod
    def _debug_printing_and_displaying_predictions(coco, coco_res, data_source, should_display_debug_images):
        for coco_data_el in coco_res.dataset['annotations']:
            cur_name_from_dataset = coco_data_el.get('_image_name_from_dataset', None)
            x1, y1, w, h = coco_data_el['bbox']
            x2 = x1+w
            y2 = y1+h
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            category_id = coco_data_el['category_id']
            category_name = coco.cats[category_id]['name']

            coco_image_id = coco_data_el['image_id']
            cur_name = coco.imgs[coco_image_id]['file_name']
            assert cur_name == cur_name_from_dataset or cur_name_from_dataset is None

            s = coco_data_el['score']

            print_info("cur_name =" + cur_name)
            print_info("        {} {} {} {}    {} %    {}".format(
                x1, y1, x2, y2, int(100*s), category_name))
            if should_display_debug_images:
                img_path = os.path.join(str(data_source), str(cur_name))
                img = cv2.imread(img_path)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.imshow("img", img)
                key = 0
                while key not in (32, 27):
                    key = cv2.waitKey() & 0xff
                should_display_debug_images = (key != 27)

    @staticmethod
    def _run_coco_evaluation(coco, coco_res, iou_type='bbox', threshold=None):
        if isinstance(coco_eval, UnsupportedPackage):
            coco_eval.raise_error("MSCOCOorigBaseMetric")
        cocoeval = coco_eval(coco, coco_res, iouType=iou_type)
        if threshold is not None:
            cocoeval.params.iouThrs = threshold
        cocoeval.evaluate()
        cocoeval.accumulate()
        cocoeval.summarize()
        res = cocoeval.stats.tolist()
        res_len = len(res)
        middle_index = res_len //2
        assert res_len == 12 if iou_type != 'keypoints' else 10

        res = [res[:middle_index], res[middle_index:]]

        return res

    @property
    def _use_original_coco(self):
        subsample_size = self.dataset.config.get('subsample_size')
        if not subsample_size:
            annotation_conversion_parameters = self.dataset.config.get('annotation_conversion')
            if annotation_conversion_parameters:
                annotation_file = annotation_conversion_parameters.get('annotation_file')
                return annotation_file.is_file()

        return False

    def compute_precision_recall(self, annotations, predictions):
        if self._use_original_coco:
            structures = self._prepare_original_coco_structures()
            coco, map_coco_img_file_name_to_img_id, map_pred_label_id_to_coco_cat_id = structures

            coco_data_to_store = self._convert_data_to_original_coco_format(
                predictions, map_coco_img_file_name_to_img_id, map_pred_label_id_to_coco_cat_id
            )
        else:
            coco = self._prepare_generated_coco_structures(annotations)
            coco_data_to_store = self._convert_data_to_generated_coco_format(predictions)

        if not coco_data_to_store:
            warnings.warn("No detections to compute coco_orig_metric")
            return [[0], [0]]

        coco_res = self._reload_results_to_coco_class(coco, coco_data_to_store)

        if SHOULD_SHOW_PREDICTIONS:
            data_source = self.dataset.config.get('data_source')
            should_display_debug_images = SHOULD_DISPLAY_DEBUG_IMAGES
            self._debug_printing_and_displaying_predictions(coco, coco_res, data_source, should_display_debug_images)

        res = self._run_coco_evaluation(coco, coco_res, self.iou_type, self.threshold)
        print_info("MSCOCOorigBaseMetric.compute_precision_recall: returning " + str(res))

        return res

    @staticmethod
    def _evaluate_image(
            ground_truth, gt_difficult, iscrowd, detections, dt_difficult, scores, iou, thresholds, profile=False
    ):
        thresholds_num = len(thresholds)
        gt_num = len(ground_truth)
        dt_num = len(detections)
        gt_matched = np.zeros((thresholds_num, gt_num))
        dt_matched = np.zeros((thresholds_num, dt_num))
        gt_ignored = gt_difficult
        dt_ignored = np.zeros((thresholds_num, dt_num))
        if np.size(iou):
            for tind, t in enumerate(thresholds):
                for dtind, _ in enumerate(detections):
                    # information about best match so far (matched_id = -1 -> unmatched)
                    iou_current = min([t, 1 - 1e-10])
                    matched_id = -1
                    for gtind, _ in enumerate(ground_truth):
                        # if this gt already matched, and not a crowd, continue
                        if gt_matched[tind, gtind] > 0 and not iscrowd[gtind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if matched_id > -1 and not gt_ignored[matched_id] and gt_ignored[gtind]:
                            break
                        # continue to next gt unless better match made
                        if iou[dtind, gtind] < iou_current:
                            continue
                        # if match successful and best so far, store appropriately
                        iou_current = iou[dtind, gtind]
                        matched_id = gtind
                    # if match made store id of match for both dt and gt
                    if matched_id == -1:
                        continue
                    dt_ignored[tind, dtind] = gt_ignored[matched_id]
                    dt_matched[tind, dtind] = 1
                    gt_matched[tind, matched_id] = dtind
        # store results for given image
        results = {
            'dt_matches': dt_matched,
            'gt_matches': gt_matched,
            'gt_ignore': gt_ignored,
            'dt_ignore': np.logical_or(dt_ignored, dt_difficult),
            'scores': scores
        }
        if profile:
            results.update({
                'iou': iou
            })

        return results

    def evaluate(self, annotations, predictions):
        pass


class MSCOCOorigAveragePrecision(MSCOCOorigBaseMetric):
    __provider__ = 'coco_orig_precision'

    def evaluate(self, annotations, predictions):
        if self.profiler:
            self.profiler.finish()
        return self.compute_precision_recall(annotations, predictions)[0][0]


class MSCOCOOrigSegmAveragePrecision(MSCOCOorigAveragePrecision, PerImageEvaluationMetric): # pylint:disable=R0901
    __provider__ = 'coco_orig_segm_precision'
    annotation_types = (CoCoInstanceSegmentationAnnotation, )
    prediction_types = (CoCocInstanceSegmentationPrediction, )

    iou_type = 'segm'

    def update(self, annotation, prediction):
        if self.profiler:
            per_class_matching = {}
            for _, label in enumerate(self.labels):
                detections, scores, dt_difficult = self._prepare_predictions(prediction, label)
                ground_truth, gt_difficult, iscrowd = self._prepare_annotations(annotation, label)
                if not ground_truth.size:
                    continue
                iou = self._compute_iou(ground_truth, detections, iscrowd)
                eval_result = self._evaluate_image(
                    ground_truth, gt_difficult, iscrowd, detections, dt_difficult, scores, iou, self.threshold,
                    True
                )
                eval_result['gt'] = annotation.to_polygon()[label]
                eval_result['dt'] = annotation.to_polygon()[label]
                per_class_matching[label] = eval_result
            per_class_result = {k: compute_precision_recall(
                self.threshold, [v])[0] for k, v in per_class_matching.items()
                                }
            for label in per_class_matching:
                per_class_matching[label]['result'] = per_class_result[label]
            self.profiler.update(
                annotation.identifier, per_class_matching, self.name, np.nanmean(list(per_class_result.values()))
            )

    @staticmethod
    def _compute_iou(gt, dets, iscrowd):
        return iou_calc(list(dets), list(gt), iscrowd)


    @staticmethod
    def _prepare_predictions(prediction, label):
        if prediction.size == 0:
            return [], [], []
        prediction_ids = np.argwhere(prediction.labels == label).reshape(-1)
        scores = prediction.scores[prediction_ids]
        if np.size(scores) == 0:
            return [], [], []
        scores_ids = np.argsort(- scores, kind='mergesort')
        difficult_mask = np.full(prediction.size, False)
        difficult_mask[prediction.metadata.get('difficult_boxes', [])] = True
        difficult_for_label = difficult_mask[prediction_ids]
        detections = [prediction.mask[idx] for idx in prediction_ids]
        detections = np.array(detections)[scores_ids]

        return detections, scores[scores_ids], difficult_for_label[scores_ids]

    @staticmethod
    def _prepare_annotations(annotation, label):
        annotation_ids = np.argwhere(np.array(annotation.labels) == label).reshape(-1)
        difficult_mask = np.full(annotation.size, False)
        difficult_indices = annotation.metadata.get("difficult_boxes", [])
        iscrowd = np.array(annotation.metadata.get('iscrowd', [0] * annotation.size))
        difficult_mask[difficult_indices] = True
        difficult_mask[iscrowd > 0] = True
        difficult_label = difficult_mask[annotation_ids]
        not_difficult_box_indices = np.argwhere(~difficult_label).reshape(-1)
        difficult_box_indices = np.argwhere(difficult_label).reshape(-1)
        iscrowd_label = iscrowd[annotation_ids]
        order = np.hstack((not_difficult_box_indices, difficult_box_indices)).astype(int)
        ann = np.array([annotation.mask[idx] for idx in annotation_ids])

        return ann[order], difficult_label[order], iscrowd_label[order]


    @staticmethod
    def _iou_type_data_to_coco(data_to_store, data, box_side_delta):
        encoded_masks = data.mask

        for data_record, segm_mask in zip(data_to_store, encoded_masks):
            data_record.update({'segmentation': deepcopy(segm_mask)})

        return data_to_store

    def _iou_type_specific_coco_annotation(self, annotation_data_to_store, annotation):
        encoded_masks = annotation.mask

        for data_record, area, segm_mask in zip(
                annotation_data_to_store, annotation.areas, encoded_masks
        ):

            mask = deepcopy(segm_mask)
            mask.update({'counts': str(mask.get('counts'), 'utf-8')})
            data_record.update({
                'area': float(area),
                'segmentation': mask
            })

        return annotation_data_to_store



class MSCOCOorigRecall(MSCOCOorigBaseMetric):
    __provider__ = 'coco_orig_recall'

    def evaluate(self, annotations, predictions):
        if self.profiler:
            self.profiler.finish()
        return self.compute_precision_recall(annotations, predictions)[1][2]


class MSCOCOorigSegmRecall(MSCOCOorigRecall):
    __provider__ = 'coco_orig_segm_recall'
    annotation_types = (CoCoInstanceSegmentationAnnotation, )
    prediction_types = (CoCocInstanceSegmentationPrediction, )

    iou_type = 'segm'

    @staticmethod
    def _iou_type_data_to_coco(data_to_store, data, box_side_delta):
        encoded_masks = data.mask

        for data_record, segm_mask in zip(data_to_store, encoded_masks):
            data_record.update({'segmentation': deepcopy(segm_mask)})

        return data_to_store

    def _iou_type_specific_coco_annotation(self, annotation_data_to_store, annotation):
        for data_record, area, segm_mask in zip(
                annotation_data_to_store, annotation.areas, annotation.mask
        ):
            mask = deepcopy(segm_mask)
            mask.update({'counts': str(mask.get('counts'), 'utf-8')})
            data_record.update({
                'area': float(area),
                'segmentation': mask
            })

        return annotation_data_to_store


class MSCOCOOrigKeyPointsAveragePrecision(MSCOCOorigAveragePrecision):
    __provider__ = 'coco_orig_keypoints_precision'
    annotation_types = (PoseEstimationAnnotation, )
    prediction_types = (PoseEstimationPrediction, )

    iou_type = 'keypoints'

    @staticmethod
    def _iou_type_data_to_coco(data_to_store, data, box_side_delta):
        for data_record, x_val, y_val, vis in zip(
                data_to_store, data.x_values, data.y_values, data.visibility
        ):
            keypoints = []
            for x, y, v in zip(x_val, y_val, vis):
                keypoints.extend([x, y, int(v)])
            data_record.update({
                'keypoints': keypoints
            })

        return data_to_store

    def _iou_type_specific_coco_annotation(self, annotation_data_to_store, annotation):
        bboxes = []
        if annotation.metadata.get('rects'):
            for rect in annotation.metadata.get('rects'):
                bboxes.append(rect)
        else:
            for x_val, y_val in zip(annotation.x_values, annotation.y_values):
                x_min = np.min(x_val)
                x_max = np.max(x_val)
                y_min = np.min(x_val)
                y_max = np.max(x_val)
                bboxes.append([x_min, y_min, x_max-x_min, y_max-y_min])

        for data_record, area, bbox, x_val, y_val, vis in zip(
                annotation_data_to_store, annotation.areas,
                bboxes, annotation.x_values, annotation.y_values, annotation.visibility
        ):
            keypoints = []
            num_keypoints = 0
            for x, y, v in zip(x_val, y_val, vis):
                keypoints.extend([int(x), int(y), int(v)])
                if v != 0:
                    num_keypoints += 1
            data_record.update({
                'bbox': [float(bb) for bb in bbox],
                'area': float(area),
                'num_keypoints': num_keypoints,
                'keypoints': keypoints
            })
        return annotation_data_to_store
