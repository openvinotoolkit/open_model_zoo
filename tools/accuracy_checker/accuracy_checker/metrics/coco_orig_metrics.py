"""
Copyright (c) 2019 Intel Corporation

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
import warnings
from pathlib import Path
try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None
try:
    from pycocotools.cocoeval import COCOeval as coco_eval
except ImportError:
    coco_eval = None
from ..representation import (
    DetectionPrediction,
    DetectionAnnotation,
    CoCoInstanceSegmentationAnnotation,
    CoCocInstanceSegmentationPrediction,
    PoseEstimationAnnotation,
    PoseEstimationPrediction
)
from ..logging import print_info
from ..config import BaseField, ConfigError
from ..utils import get_or_parse_value
from .metric import FullDatasetEvaluationMetric
from .coco_metrics import COCO_THRESHOLDS

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

        return parameters

    def configure(self):
        self.threshold = get_or_parse_value(self.get_value_from_config('threshold'), COCO_THRESHOLDS)
        if not self.dataset.metadata:
            raise ConfigError('coco orig metrics require dataset_meta'
                              'Please provide dataset meta file or regenerate annotation')

        if not self.dataset.metadata.get('label_map'):
            raise ConfigError('coco_orig metrics require label_map providing in dataset_meta'
                              'Please provide dataset meta file or regenerated annotation')

    @staticmethod
    def _iou_type_data_to_coco(data_to_store, data):
        x_mins = data.x_mins.tolist()
        y_mins = data.y_mins.tolist()
        x_maxs = data.x_maxs.tolist()
        y_maxs = data.y_maxs.tolist()

        for data_record, x_min, y_min, x_max, y_max in zip(
                data_to_store, x_mins, y_mins, x_maxs, y_maxs
        ):
            width = x_max - x_min + 1
            height = y_max - y_min + 1
            area = width * height
            data_record.update({
                'bbox': [x_min, y_min, width, height],
                'area': area
            })

        return data_to_store

    def _prepare_coco_structures(self, annotations):
        label_map = self.dataset.metadata.get('label_map')
        if self.dataset.metadata.get('background_label') is not None:
            label_map.pop(self.dataset.metadata.get('background_label'))

        if COCO is None:
            raise ValueError('pycocotools is not installed, please install it')
        coco_data_to_store = self._prepare_data_for_annotation_file(
            annotations, label_map)

        coco_annotation = self._create_json(coco_data_to_store)
        coco = COCO(str(coco_annotation))

        return coco

    def _convert_data_to_coco_format(self, predictions):
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
            prediction_data_to_store = self._iou_type_data_to_coco(prediction_data_to_store, pred)
            coco_data_to_store.extend(prediction_data_to_store)

        return coco_data_to_store

    def _iou_type_specific_coco_annotation(self, annotation_data_to_store, annotation):
        annotation_data_to_store = self._iou_type_data_to_coco(annotation_data_to_store, annotation)
        return annotation_data_to_store

    def _prepare_data_for_annotation_file(
            self, annotations, dataset_label_map
    ):
        coco_annotation_to_store = []
        coco_category_to_store = []
        coco_image_to_store = []
        count = 0
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

            coco_image_to_store.append({
                'id': cur_img_id,
                'file_name': cur_name,
                'width': annotation.metadata.get('image_size')[0][0],
                'height': annotation.metadata.get('image_size')[0][1]
            })

        for cat, cat_name in dataset_label_map.items():
            coco_category_to_store.append({
                'id': cat,
                'name': cat_name
            })

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
        if coco_eval is None:
            raise ValueError('pycocotools is not installed, please install it before usage')
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

    def compute_precision_recall(self, annotations, predictions):
        coco = self._prepare_coco_structures(annotations)
        coco_data_to_store = self._convert_data_to_coco_format(predictions)

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

    def evaluate(self, annotations, predictions):
        pass


class MSCOCOorigAveragePrecision(MSCOCOorigBaseMetric):
    __provider__ = 'coco_orig_precision'

    def evaluate(self, annotations, predictions):
        return self.compute_precision_recall(annotations, predictions)[0][0]


class MSCOCOOrigSegmAveragePrecision(MSCOCOorigAveragePrecision):
    __provider__ = 'coco_orig_segm_precision'
    annotation_types = (CoCoInstanceSegmentationAnnotation, )
    prediction_types = (CoCocInstanceSegmentationPrediction, )

    iou_type = 'segm'

    @staticmethod
    def _iou_type_data_to_coco(data_to_store, data):
        encoded_masks = data.mask

        for data_record, segm_mask in zip(data_to_store, encoded_masks):
            data_record.update({'segmentation': segm_mask})

        return data_to_store

    def _iou_type_specific_coco_annotation(self, annotation_data_to_store, annotation):
        encoded_masks = annotation.mask

        for data_record, area, segm_mask in zip(
                annotation_data_to_store, annotation.areas, encoded_masks
        ):
            segm_mask.update({'counts': str(segm_mask.get('counts'), 'utf-8')})
            data_record.update({
                'area': float(area),
                'segmentation': segm_mask
            })

        return annotation_data_to_store


class MSCOCOorigRecall(MSCOCOorigBaseMetric):
    __provider__ = 'coco_orig_recall'

    def evaluate(self, annotations, predictions):
        return self.compute_precision_recall(annotations, predictions)[1][2]


class MSCOCOorigSegmRecall(MSCOCOorigRecall):
    __provider__ = 'coco_orig_segm_recall'
    annotation_types = (CoCoInstanceSegmentationAnnotation, )
    prediction_types = (CoCocInstanceSegmentationPrediction, )

    iou_type = 'segm'

    @staticmethod
    def _iou_type_data_to_coco(data_to_store, data):
        encoded_masks = data.mask

        for data_record, segm_mask in zip(data_to_store, encoded_masks):
            data_record.update({'segmentation': segm_mask})

        return data_to_store

    def _iou_type_specific_coco_annotation(self, annotation_data_to_store, annotation):
        encoded_masks = annotation.mask

        for data_record, area, segm_mask in zip(
                annotation_data_to_store, annotation.areas, encoded_masks
        ):
            segm_mask.update({'counts': str(segm_mask.get('counts'), 'utf-8')})
            data_record.update({
                'area': float(area),
                'segmentation': segm_mask
            })

        return annotation_data_to_store


class MSCOCOOrigKeyPointsAveragePrecision(MSCOCOorigAveragePrecision):
    __provider__ = 'coco_orig_keypoints_precision'
    annotation_types = (PoseEstimationAnnotation, )
    prediction_types = (PoseEstimationPrediction, )

    iou_type = 'keypoints'

    @staticmethod
    def _iou_type_data_to_coco(data_to_store, data):
        for data_record, x_val, y_val, vis in zip(
                data_to_store, data.x_values, data.y_values, data.visibility
        ):
            keypoints = []
            num_keypoints = 0
            for x, y, v in zip(x_val, y_val, vis):
                keypoints.extend([int(x), int(y), int(v)])
                if x != 0:
                    num_keypoints += 1
            data_record.update({
                'keypoints': keypoints,
                'num_keypoints': num_keypoints
            })

        return data_to_store

    def _iou_type_specific_coco_annotation(self, annotation_data_to_store, annotation):
        annotation_data_to_store = self._iou_type_data_to_coco(annotation_data_to_store, annotation)
        for data_record, bbox, area in zip(
                annotation_data_to_store, annotation.bboxes, annotation.areas
        ):
            data_record.update({
                'bbox': [float(bb) for bb in bbox],
                'area': float(area)
            })
        return annotation_data_to_store
