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

from ..representation import (
    DetectionPrediction,
    DetectionAnnotation,
    CoCoInstanceSegmentationAnnotation,
    CoCocInstanceSegmentationPrediction,
    PoseEstimationAnnotation,
    PoseEstimationPrediction
)
from ..logging import print_info
from ..config import BaseField
from ..utils import get_or_parse_value
from .metric import FullDatasetEvaluationMetric
from .coco_metrics import COCO_THRESHOLDS

SHOULD_SHOW_PREDICTIONS = False
SHOULD_DISPLAY_DEBUG_IMAGES = False
if SHOULD_DISPLAY_DEBUG_IMAGES:
    import cv2

def box_to_coco(prediction_data_to_store, pred):
    x_mins = pred.x_mins.tolist()
    y_mins = pred.y_mins.tolist()
    x_maxs = pred.x_maxs.tolist()
    y_maxs = pred.y_maxs.tolist()

    for data_record, x_min, y_min, x_max, y_max in zip(
            prediction_data_to_store, x_mins, y_mins, x_maxs, y_maxs
    ):
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        data_record.update({'bbox': [x_min, y_min, width, height]})

    return prediction_data_to_store

def segm_to_coco(prediction_data_to_store, pred):
    encoded_masks = pred.mask

    for data_record, segm_mask in zip(prediction_data_to_store, encoded_masks):
        data_record.update({'segmentation': segm_mask})

    return prediction_data_to_store


def keypoints_to_coco(prediction_data_to_store, pred):
    for data_record, x_val, y_val, vis in zip(
            prediction_data_to_store, pred.x_values, pred.y_values, pred.visibility
    ):
        keypoints = []
        for x, y, v in zip(x_val, y_val, vis):
            keypoints.extend([x, y, int(v)])
        data_record.update({
            'keypoints': keypoints
        })

    return prediction_data_to_store

iou_specific_processing = {
    'bbox': box_to_coco,
    'segm': segm_to_coco,
    'keypoints': keypoints_to_coco
}


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

    @staticmethod
    def generate_map_pred_label_id_to_coco_cat_id(has_background, use_full_label_map):
        shift = 0 if has_background else 1
        max_cat = 90 if use_full_label_map else 80
        max_key = max_cat - shift
        res_map = {i: i + shift for i in range(0, max_key+1)}
        assert max(res_map.values()) == max_cat
        return res_map

    def _prepare_coco_structures(self):
        from pycocotools.coco import COCO

        annotation_conversion_parameters = self.dataset.config.get('annotation_conversion')
        if not annotation_conversion_parameters:
            raise ValueError('annotation_conversion parameter is not pointed, '
                             'but it is required for coco original metrics')
        annotation_file = annotation_conversion_parameters.get('annotation_file')
        if not annotation_file.is_file():
            raise ValueError("annotation file '{}' is not found".format(annotation_file))
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

    @staticmethod
    def _convert_data_to_coco_format(
            predictions, map_coco_img_file_name_to_img_id, map_pred_label_id_to_coco_cat_id, iou_type='bbox'
    ):
        coco_data_to_store = []
        for pred in predictions:
            prediction_data_to_store = []
            cur_name = pred.identifier
            cur_name = os.path.basename(cur_name)
            assert cur_name in map_coco_img_file_name_to_img_id
            cur_img_id = map_coco_img_file_name_to_img_id[cur_name]

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
            iou_specific_converter = iou_specific_processing.get(iou_type)
            if iou_specific_converter is None:
                raise ValueError("unknown iou type: '{}'".format(iou_type))
            prediction_data_to_store = iou_specific_converter(prediction_data_to_store, pred)
            coco_data_to_store.extend(prediction_data_to_store)

        return coco_data_to_store

    @staticmethod
    def _reload_results_to_coco_class(coco, coco_data_to_store):
        with tempfile.NamedTemporaryFile() as ftmp:
            json_file_to_store = ftmp.name + ".json"
        with open(json_file_to_store, 'w') as f:
            json.dump(coco_data_to_store, f, indent=4)

        json_file_to_load = json_file_to_store
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
        from pycocotools.cocoeval import COCOeval

        cocoeval = COCOeval(coco, coco_res, iouType=iou_type)
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

    def compute_precision_recall(self, predictions):
        coco, map_coco_img_file_name_to_img_id, map_pred_label_id_to_coco_cat_id = self._prepare_coco_structures()

        coco_data_to_store = self._convert_data_to_coco_format(
            predictions, map_coco_img_file_name_to_img_id, map_pred_label_id_to_coco_cat_id, self.iou_type
        )

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
        return self.compute_precision_recall(predictions)[0][0]


class MSCOCOOrigSegmAveragePrecision(MSCOCOorigAveragePrecision):
    __provider__ = 'coco_orig_segm_precision'
    annotation_types = (CoCoInstanceSegmentationAnnotation, )
    prediction_types = (CoCocInstanceSegmentationPrediction, )

    iou_type = 'segm'


class MSCOCOorigRecall(MSCOCOorigBaseMetric):
    __provider__ = 'coco_orig_recall'

    def evaluate(self, annotations, predictions):
        return self.compute_precision_recall(predictions)[1][2]


class MSCOCOorigSegmRecall(MSCOCOorigRecall):
    __provider__ = 'coco_orig_segm_recall'
    annotation_types = (CoCoInstanceSegmentationAnnotation, )
    prediction_types = (CoCocInstanceSegmentationPrediction, )

    iou_type = 'segm'


class MSCOCOOrigKeyPointsAveragePrecision(MSCOCOorigAveragePrecision):
    __provider__ = 'coco_orig_keypoints_precision'
    annotation_types = (PoseEstimationAnnotation, )
    prediction_types = (PoseEstimationPrediction, )

    iou_type = 'keypoints'
