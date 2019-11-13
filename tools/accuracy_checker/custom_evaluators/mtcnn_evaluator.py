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

import copy
import numpy as np
import cv2

from accuracy_checker.evaluators import BaseEvaluator
from accuracy_checker.adapters import create_adapter
from accuracy_checker.launcher import create_launcher, InputFeeder
from accuracy_checker.dataset import Dataset
from accuracy_checker.data_readers import BaseReader
from accuracy_checker.preprocessor import PreprocessingExecutor
from accuracy_checker.utils import extract_image_representations
from accuracy_checker.adapters import MTCNNPAdapter
from accuracy_checker.metrics import MetricsExecutor
from accuracy_checker.postprocessor import PostprocessingExecutor
from accuracy_checker.config import ConfigError


class BaseStage:
    def __init__(self, model_info, input_feeder, preprocessor):
        self.model_info = model_info
        self.input_feeder = input_feeder
        self.preprocessor = preprocessor

    def predict(self, input_blobs, batch_meta):
        raise NotImplementedError

    def preprocess_data(self, batch_input, batch_annotation, previous_stage_prediction, *args, **kwargs):
        raise NotImplementedError

    def postprocess_result(self, identifiers, this_stage_result, batch_meta, previous_stage_result, *args, **kwargs):
        raise NotImplementedError


class ProposalBaseStage(BaseStage):
    def __init__(self, model_info, input_feeder, preprocessor, adapter):
        super().__init__(model_info, input_feeder, preprocessor)
        self.adapter = adapter

    def preprocess_data(self, batch_input, batch_annotation, *args, **kwargs):
        batch_input = self.preprocessor.process(batch_input, batch_annotation)
        _, batch_meta = extract_image_representations(batch_input)
        filled_inputs = self.input_feeder.fill_inputs(batch_input)
        return filled_inputs, batch_meta

    def postprocess_result(self, identifiers, this_stage_result, batch_meta, *args, **kwargs):
        return self.adapter.process(this_stage_result, identifiers, batch_meta)

    def _infer(self, input_blobs, batch_meta):
        raise NotImplementedError

    def predict(self, input_blobs, batch_meta):
        return self._infer(input_blobs, batch_meta)


class RefineBaseStage(BaseStage):
    input_size = 24
    include_boundaries = True

    def preprocess_data(self, batch_input, batch_annotation, previous_stage_prediction, *lrgs, **kwargs):
        batch_input = self.preprocessor.process(batch_input, batch_annotation)
        _, batch_meta = extract_image_representations(batch_input)
        batch_input = [
            cut_roi(input_image, prediction, self.input_size, include_bound=self.include_boundaries)
            for input_image, prediction in zip(batch_input, previous_stage_prediction)
        ]
        filled_inputs = self.input_feeder.fill_inputs(batch_input)
        return filled_inputs, batch_meta

    def postprocess_result(self, identifiers, this_stage_result, batch_meta, previous_stage_result, *args, **kwargs):
        return calibrate_predictions(
            previous_stage_result, this_stage_result, 0.7, self.model_info['outputs'], 'Union'
        )

    def _infer(self, input_blobs, batch_meta):
        raise NotImplementedError

    def predict(self, input_blobs, batch_meta):
        return self._infer(input_blobs, batch_meta)


class OutputStage(BaseStage):
    input_size = 48
    include_boundaries = False

    def postprocess_result(self, identifiers, this_stage_result, batch_meta, previous_stage_result, *args, **kwargs):
        batch_predictions = calibrate_predictions(
            this_stage_result, previous_stage_result, 0.7, self.model_info['outputs']
        )
        batch_predictions[0], _ = nms(batch_predictions[0], 0.7, 'Min')

        return batch_predictions

class CaffeProposalStage(ProposalBaseStage):
    def __init__(self,  model_info, input_feeder, preprocessor, adapter, launcher):
        super().__init__(model_info, input_feeder, preprocessor, adapter)
        self.net =


class MTCNNEvaluator(BaseEvaluator):
    def __init__(
            self, dataset, reader, input_feeders, launcher_stage1, launcher_stage2, launcher_stage3,
            metrics_executor, preprocessing_executors, first_stage_adapter,
            models_info, postprocessing
    ):
        super().__init__()
        self.dataset = dataset
        self.reader = reader
        self.launcher_stage1 = launcher_stage1
        self.launcher_stage2 = launcher_stage2
        self.launcher_stage3 = launcher_stage3
        self.preprocessing_executors = preprocessing_executors
        self.metrics_executor = metrics_executor
        self.input_feeders = input_feeders
        self.adapter = first_stage_adapter
        self.models_info = models_info
        self.postprocessing = postprocessing
        self._metrics_results = []
        self._annotations, self._predictions = [], []

    def process_dataset(self, stored_predictions, progress_reporter, *args, **kwargs):
        if progress_reporter:
            progress_reporter.reset(self.dataset.size)
        for batch_id, batch_annotation in enumerate(self.dataset):
            batch_identifiers = [annotation.identifier for annotation in batch_annotation]
            batch_input = [self.reader(identifier=identifier) for identifier in batch_identifiers]
            batch_input = self.preprocessing_executors[0].process(batch_input, batch_annotation)
            _, batch_meta = extract_image_representations(batch_input)
            filled_inputs = self.input_feeders[0].fill_inputs(batch_input)
            batch_predictions = self.launcher_stage1.predict(filled_inputs, batch_meta, **kwargs)
            batch_predictions = self.adapter.process(batch_predictions, batch_identifiers, batch_meta)
            batch_input = [self.reader(identifier=identifier) for identifier in batch_identifiers]
            batch_input = self.preprocessing_executors[1].process(batch_input)
            batch_input = [
                cut_roi(input_image, prediction, 24, include_bound=True)
                for input_image, prediction in zip(batch_input, batch_predictions)
            ]
            filled_inputs = self.input_feeders[1].fill_inputs(batch_input)
            batch_s2_predictions = self.launcher_stage2.predict(filled_inputs, batch_meta, **kwargs)
            batch_predictions = calibrate_predictions(
                batch_predictions, batch_s2_predictions, 0.7, self.models_info['rnet']['outputs'], 'Union'
            )
            batch_input = [self.reader(identifier=identifier) for identifier in batch_identifiers]
            self.preprocessing_executors[2].process(batch_input)
            batch_input = [
                cut_roi(input_image, prediction, 48, include_bound=False)
                for input_image, prediction in zip(batch_input, batch_predictions)
            ]
            if np.size(batch_input[0].data):
                filled_inputs = self.input_feeders[2].fill_inputs(batch_input)
                batch_s3_predictions = self.launcher_stage3.predict(filled_inputs, batch_meta, **kwargs)
                batch_predictions = calibrate_predictions(
                    batch_predictions, batch_s3_predictions, 0.7, self.models_info['onet']['outputs']
                )
                batch_predictions[0], _ = nms(batch_predictions[0], 0.7, 'Min')
                batch_annotation, batch_predictions = self.postprocessing.process_batch(
                    batch_annotation, batch_predictions
                )

            self._annotations.extend(batch_annotation)
            self._predictions.extend(batch_predictions)
            if progress_reporter:
                progress_reporter.update(batch_id, len(batch_predictions))

    def compute_metrics(self, print_results=True, ignore_results_formatting=False):
        if self._metrics_results:
            del self._metrics_results
            self._metrics_results = []

        for result_presenter, evaluated_metric in self.metrics_executor.iterate_metrics(
                self._annotations, self._predictions):
            self._metrics_results.append(evaluated_metric)
            if print_results:
                result_presenter.write_result(evaluated_metric, ignore_results_formatting)

        return self._metrics_results

    @classmethod
    def from_configs(cls, config):
        def update_mo_params(launcher_config, model_config):
            if 'mo_params' in launcher_config and 'mo_params' in model_config:
                for key, value in launcher_config['mo_params'].items():
                    model_config['mo_params'][key] = value

            if 'mo_flags' in launcher_config and 'mo_flags' in model_config:
                common_mo_flags = model_config['mo_flags']
                for flag in launcher_config['mo_flags']:
                    if flag not in common_mo_flags:
                        common_mo_flags.append(flag)

        def merge_preprocessing(model_specific, common_preprocessing):
            if model_specific:
                model_specific.extend(common_preprocessing)
                return model_specific
            return common_preprocessing

        dataset_config = config['datasets'][0]
        dataset = Dataset(dataset_config)
        data_reader_config = dataset_config.get('reader', 'opencv_imread')
        data_source = dataset_config['data_source']
        if isinstance(data_reader_config, str):
            reader = BaseReader.provide(data_reader_config, data_source)
        elif isinstance(data_reader_config, dict):
            reader = BaseReader.provide(data_reader_config['type'], data_source, data_reader_config)
        else:
            raise ConfigError('reader should be dict or string')
        models_info = config['networks_info']
        pnet_config = models_info['pnet']
        rnet_config = models_info['rnet']
        onet_config = models_info['onet']
        launcher_specific_entries = [
            'model', 'weights', 'caffe_model', 'caffe_weights', 'tf_model',
            'mo_params', 'mo_flags', 'inputs'
        ]
        pnet_launcher_options = {
            launcher_specific_entry: pnet_config.get(launcher_specific_entry)
            for launcher_specific_entry in launcher_specific_entries if pnet_config.get(launcher_specific_entry)
        }
        pnet_outs = pnet_config['outputs']
        pnet_adapter_config = pnet_config.get('adapter', {'type': 'mtcnn_p', **pnet_outs})
        if config['launchers'][0]['framework'] == 'caffe':
            pnet_adapter_config.update({'regions_format': 'hw'})
        rnet_launcher_options = {
            launcher_specific_entry: rnet_config.get(launcher_specific_entry)
            for launcher_specific_entry in launcher_specific_entries if rnet_config.get(launcher_specific_entry)
        }
        onet_launcher_options = {
            launcher_specific_entry: onet_config.get(launcher_specific_entry)
            for launcher_specific_entry in launcher_specific_entries if onet_config.get(launcher_specific_entry)
        }
        launcher_config_1 = config['launchers'][0]
        update_mo_params(launcher_config_1, pnet_launcher_options)
        launcher_config_1.update({'allow_reshape_input': True})
        launcher_config_1.update(pnet_launcher_options)
        launcher_config_2 = copy.deepcopy(launcher_config_1)
        launcher_config_3 = copy.deepcopy(launcher_config_1)
        update_mo_params(launcher_config_2, rnet_launcher_options)
        update_mo_params(launcher_config_3, onet_launcher_options)
        launcher_s1 = create_launcher(launcher_config_1)
        pnet_adapter = create_adapter(pnet_adapter_config, launcher_s1, dataset)
        launcher_config_2.update(rnet_launcher_options)
        launcher_s2 = create_launcher(launcher_config_2)
        launcher_config_3.update(onet_launcher_options)
        launcher_s3 = create_launcher(launcher_config_3)
        input_feeders = [
            InputFeeder(launcher_config_1['inputs'], launcher_s1.inputs, launcher_s1.fit_to_input),
            InputFeeder(launcher_config_2['inputs'], launcher_s2.inputs, launcher_s2.fit_to_input),
            InputFeeder(launcher_config_3['inputs'], launcher_s3.inputs, launcher_s3.fit_to_input)
        ]
        preprocessors = dataset_config.get('preprocessing', [])
        first_stage_preprocessing = merge_preprocessing(pnet_config.get('preprocessing', []), preprocessors)
        second_stage_preprocessing = merge_preprocessing(rnet_config.get('preprocessing', []), preprocessors)
        third_stage_preprocessing = merge_preprocessing(onet_config.get('preprocessing', []), preprocessors)
        preprocessors = [
            PreprocessingExecutor(first_stage_preprocessing, dataset.name),
            PreprocessingExecutor(second_stage_preprocessing, dataset.name),
            PreprocessingExecutor(third_stage_preprocessing, dataset.name)
        ]
        metrics_executor = MetricsExecutor(dataset_config['metrics'], dataset)
        postprocessing = PostprocessingExecutor(dataset_config['postprocessing'])
        return cls(
            dataset, reader, input_feeders, launcher_s1, launcher_s2, launcher_s3,
            metrics_executor, preprocessors, pnet_adapter, models_info, postprocessing
        )

    def release(self):
        self.launcher_stage1.release()
        self.launcher_stage2.release()
        self.launcher_stage3.release()

    def reset(self):
        self.metrics_executor.reset()
        self.dataset.reset()


def calibrate_predictions(previous_stage_predictions, out, threshold, outputs_mapping, iou_type=None):
    score = out[0][outputs_mapping['probability_out']][:, 1]
    pass_t = np.where(score > 0.7)[0]
    removed_boxes = [i for i in range(previous_stage_predictions[0].size) if i not in pass_t]
    previous_stage_predictions[0].remove(removed_boxes)
    previous_stage_predictions[0].scores = score[pass_t]
    bboxes = np.c_[
        previous_stage_predictions[0].x_mins, previous_stage_predictions[0].y_mins,
        previous_stage_predictions[0].x_maxs, previous_stage_predictions[0].y_maxs,
        previous_stage_predictions[0].scores
    ]
    mv = out[0][outputs_mapping['region_out']][pass_t]
    if iou_type:
        previous_stage_predictions[0], peek = nms(previous_stage_predictions[0], threshold, iou_type)
        bboxes = np.c_[
            previous_stage_predictions[0].x_mins, previous_stage_predictions[0].y_mins,
            previous_stage_predictions[0].x_maxs, previous_stage_predictions[0].y_maxs,
            previous_stage_predictions[0].scores
        ]
        mv = mv[np.sort(peek).astype(int)]
    bboxes = bbreg(bboxes, mv.T)
    x_mins, y_mins, x_maxs, y_maxs, _ = bboxes.T
    previous_stage_predictions[0].x_mins = x_mins
    previous_stage_predictions[0].y_mins = y_mins
    previous_stage_predictions[0].x_maxs = x_maxs
    previous_stage_predictions[0].y_maxs = y_maxs

    return previous_stage_predictions


def nms(prediction, threshold, iou_type):
    bboxes = np.c_[
        prediction.x_mins, prediction.y_mins,
        prediction.x_maxs, prediction.y_maxs,
        prediction.scores
    ]
    peek = MTCNNPAdapter.nms(bboxes, threshold, iou_type)
    prediction.remove([i for i in range(prediction.size) if i not in peek])

    return prediction, peek


def bbreg(boundingbox, reg):
    reg = reg.T

    # calibrate bounding boxes
    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1

    bb0 = boundingbox[:, 0] + reg[:, 0] * w
    bb1 = boundingbox[:, 1] + reg[:, 1] * h
    bb2 = boundingbox[:, 2] + reg[:, 2] * w
    bb3 = boundingbox[:, 3] + reg[:, 3] * h

    boundingbox[:, 0:4] = np.array([bb0, bb1, bb2, bb3]).T

    return boundingbox


def pad(boxesA, h, w):
    boxes = boxesA.copy()

    tmph = boxes[:, 3] - boxes[:, 1] + 1
    tmpw = boxes[:, 2] - boxes[:, 0] + 1
    numbox = boxes.shape[0]

    dx = np.ones(numbox)
    dy = np.ones(numbox)
    edx = tmpw
    edy = tmph

    x = boxes[:, 0:1][:, 0]
    y = boxes[:, 1:2][:, 0]
    ex = boxes[:, 2:3][:, 0]
    ey = boxes[:, 3:4][:, 0]

    tmp = np.where(ex > w)[0]
    if tmp.shape[0] != 0:
        edx[tmp] = -ex[tmp] + w - 1 + tmpw[tmp]
        ex[tmp] = w - 1

    tmp = np.where(ey > h)[0]
    if tmp.shape[0] != 0:
        edy[tmp] = -ey[tmp] + h - 1 + tmph[tmp]
        ey[tmp] = h - 1

    tmp = np.where(x < 1)[0]
    if tmp.shape[0] != 0:
        dx[tmp] = 2 - x[tmp]
        x[tmp] = np.ones_like(x[tmp])

    tmp = np.where(y < 1)[0]
    if tmp.shape[0] != 0:
        dy[tmp] = 2 - y[tmp]
        y[tmp] = np.ones_like(y[tmp])

    # for python index from 0, while matlab from 1
    dy = np.maximum(0, dy - 1)
    dx = np.maximum(0, dx - 1)
    y = np.maximum(0, y - 1)
    x = np.maximum(0, x - 1)
    edy = np.maximum(0, edy - 1)
    edx = np.maximum(0, edx - 1)
    ey = np.maximum(0, ey - 1)
    ex = np.maximum(0, ex - 1)
    return [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]


def rerec(bboxA):
    w = bboxA[:, 2] - bboxA[:, 0]
    h = bboxA[:, 3] - bboxA[:, 1]
    l = np.maximum(w, h).T

    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
    bboxA[:, 2:4] = bboxA[:, 0:2] + np.repeat([l], 2, axis=0).T

    return bboxA


def cut_roi(image, prediction, dst_size, include_bound=True):
    bboxes = np.c_[
        prediction.x_mins, prediction.y_mins,
        prediction.x_maxs, prediction.y_maxs,
        prediction.scores
    ]
    img = image.data
    bboxes = rerec(bboxes)
    bboxes[:, 0:4] = np.fix(bboxes[:, 0:4])
    dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(bboxes, *img.shape[:2])
    numbox = bboxes.shape[0]
    tempimg = np.zeros((numbox, dst_size, dst_size, 3))
    for k in range(numbox):
        tmp_k_h = int(tmph[k]) + int(include_bound)
        tmp_k_w = int(tmpw[k]) + int(include_bound)
        tmp = np.zeros((tmp_k_h, tmp_k_w, 3))
        tmp_ys = slice(int(dy[k]), int(edy[k]) + 1)
        tmp_xs = slice(int(dx[k]), int(edx[k]) + 1)
        img_ys = slice(int(y[k]), int(ey[k]) + 1)
        img_xs = slice(int(x[k]), int(ex[k]) + 1)
        tmp[tmp_ys, tmp_xs] = img[img_ys, img_xs]
        tempimg[k, :, :, :] = cv2.resize(tmp, (dst_size, dst_size))
    image.data = tempimg

    return image
