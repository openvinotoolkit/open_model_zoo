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
from collections import OrderedDict
import pickle
from pathlib import Path
import numpy as np
import cv2

from accuracy_checker.evaluators import BaseEvaluator
from accuracy_checker.adapters import create_adapter
from accuracy_checker.launcher import create_launcher, InputFeeder
from accuracy_checker.dataset import Dataset
from accuracy_checker.data_readers import BaseReader, REQUIRES_ANNOTATIONS
from accuracy_checker.preprocessor import PreprocessingExecutor
from accuracy_checker.utils import extract_image_representations, read_pickle, contains_any
from accuracy_checker.adapters import MTCNNPAdapter
from accuracy_checker.metrics import MetricsExecutor
from accuracy_checker.postprocessor import PostprocessingExecutor
from accuracy_checker.config import ConfigError


def build_stages(models_info, preprocessors_config, launcher, model_args):
    def merge_preprocessing(model_specific, common_preprocessing):
        if model_specific:
            model_specific.extend(common_preprocessing)
            return model_specific
        return common_preprocessing

    required_stages = ['pnet']
    stages_mapping = OrderedDict([
        ('pnet', {'caffe': CaffeProposalStage, 'dlsdk': DLSDKProposalStage, 'dummy': DummyProposalStage}),
        ('rnet', {'caffe': CaffeRefineStage, 'dlsdk': DLSDKRefineStage}),
        ('onet', {'caffe': CaffeOutputStage,'dlsdk': DLSDKOutputStage})
    ])
    framework = launcher.config['framework']
    stages = []
    for stage_name, stage_classes in stages_mapping.items():
        if stage_name not in models_info:
            if stage_name not in required_stages:
                continue
            else:
                raise ConfigError('{} required for evaluation'.format(stage_name))
        model_config = models_info[stage_name]
        if 'predictions' in model_config and not model_config.get('store_predictions', False):
            stage_framework = 'dummy'
        else:
            stage_framework = framework
        if not contains_any(model_config, ['model', 'caffe_model']) and stage_framework != 'dummy':
            if model_args:
                model_config['model'] = model_args[len(stages) if len(model_args) > 1 else 0]
        stage = stage_classes.get(stage_framework)
        if not stage_classes:
            raise ConfigError('{} stage does not support {} framework'.format(stage_name, stage_framework))
        stage_preprocess = merge_preprocessing(models_info[stage_name].get('preprocessing', []), preprocessors_config)
        preprocessor = PreprocessingExecutor(stage_preprocess)
        stages.append(stage(models_info[stage_name], preprocessor, launcher))

    if not stages:
        raise ConfigError('please provide information about MTCNN pipeline stages')

    return stages


class BaseStage:
    def __init__(self, model_info, preprocessor):
        self.model_info = model_info
        self.preprocessor = preprocessor
        self.input_feeder = None
        self.store = model_info.get('store_predictions', False)
        self.predictions = []

    def predict(self, input_blobs, batch_meta):
        raise NotImplementedError

    def preprocess_data(self, batch_input, batch_annotation, previous_stage_prediction, *args, **kwargs):
        raise NotImplementedError

    def postprocess_result(self, identifiers, this_stage_result, batch_meta, previous_stage_result, *args, **kwargs):
        raise NotImplementedError

    def release(self):
        pass

    def reset(self):
        self._predictions = []

    def dump_predictions(self):
        if not hasattr(self, 'prediction_file'):
            prediction_file = Path(self.model_info.get('predictions', 'predictions.pickle'))
            self.prediction_file = prediction_file
        with self.prediction_file.open('wb') as out_file:
            pickle.dump(self._predictions, out_file)


class ProposalBaseStage(BaseStage):
    default_model_name = 'mtcnn-p'

    def __init__(self, model_info, preprocessor):
        super().__init__(model_info, preprocessor)
        self.adapter = None
        self.input_feeder = None
        self._predictions = []

    def preprocess_data(self, batch_input, batch_annotation, *args, **kwargs):
        batch_input = self.preprocessor.process(batch_input, batch_annotation)
        _, batch_meta = extract_image_representations(batch_input)
        filled_inputs = self.input_feeder.fill_inputs(batch_input) if self.input_feeder else batch_input
        return filled_inputs, batch_meta

    def postprocess_result(self, identifiers, this_stage_result, batch_meta, *args, **kwargs):
        result = self.adapter.process(this_stage_result, identifiers, batch_meta) if self.adapter else this_stage_result
        if self.store:
            self._predictions.extend(result)
        return result

    def _infer(self, input_blobs, batch_meta):
        raise NotImplementedError

    def predict(self, input_blobs, batch_meta):
        return self._infer(input_blobs, batch_meta)

    def dump_predictions(self):
        if not hasattr(self, 'prediction_file'):
            prediction_file = Path(self.model_info.get('predictions', 'pnet_predictions.pickle'))
            self.prediction_file = prediction_file
        with self.prediction_file.open('wb') as out_file:
            pickle.dump(self._predictions, out_file)


class DummyProposalStage(ProposalBaseStage):
    def __init__(self, model_info, preprocessor, *args, **kwargs):
        super().__init__(model_info, preprocessor)
        self._index = 0
        if 'predictions' not in self.model_info:
            raise ConfigError('predictions_file is not found')
        self._predictions = read_pickle(self.model_info['predictions'])
        self.iterator = 0

    def preprocess_data(self, batch_input, batch_annotation, *args, **kwargs):
        _, batch_meta = extract_image_representations(batch_input)
        return batch_input, batch_meta

    def _infer(self, input_blobs, batch_meta):
        batch_size = len(batch_meta)
        results = self._predictions[self._index:self._index+batch_size]
        self._index += batch_size
        return results

    def postprocess_result(self, identifiers, this_stage_result, batch_meta, *args, **kwargs):
        return this_stage_result


class RefineBaseStage(BaseStage):
    input_size = 24
    include_boundaries = True
    default_model_name = 'mtcnn-r'

    def preprocess_data(self, batch_input, batch_annotation, previous_stage_prediction, *lrgs, **kwargs):
        batch_input = self.preprocessor.process(batch_input, batch_annotation)
        _, batch_meta = extract_image_representations(batch_input)
        batch_input = [
            cut_roi(input_image, prediction, self.input_size, include_bound=self.include_boundaries)
            for input_image, prediction in zip(batch_input, previous_stage_prediction)
        ]
        filled_inputs = self.input_feeder.fill_inputs(batch_input) if self.input_feeder else batch_input
        return filled_inputs, batch_meta

    def postprocess_result(self, identifiers, this_stage_result, batch_meta, previous_stage_result, *args, **kwargs):
        result = calibrate_predictions(
            previous_stage_result, this_stage_result, 0.7, self.model_info['outputs'], 'Union'
        )
        if self.store:
            self._predictions.extend(result)
        return result

    def _infer(self, input_blobs, batch_meta):
        raise NotImplementedError

    def predict(self, input_blobs, batch_meta):
        return self._infer(input_blobs, batch_meta)

    def dump_predictions(self):
        if not hasattr(self, 'prediction_file'):
            prediction_file = Path(self.model_info.get('predictions', 'rnet_predictions.pickle'))
            self.prediction_file = prediction_file
        with self.prediction_file.open('wb') as out_file:
            pickle.dump(self._predictions, out_file)


class OutputBaseStage(RefineBaseStage):
    input_size = 48
    include_boundaries = False
    default_model_name = 'mtcnn-o'

    def postprocess_result(self, identifiers, this_stage_result, batch_meta, previous_stage_result, *args, **kwargs):
        batch_predictions = calibrate_predictions(
            previous_stage_result, this_stage_result, 0.7, self.model_info['outputs']
        )
        batch_predictions[0], _ = nms(batch_predictions[0], 0.7, 'Min')
        if self.store:
            self._predictions.extend(batch_predictions)
        return batch_predictions

    def dump_predictions(self):
        if not hasattr(self, 'prediction_file'):
            prediction_file = Path(self.model_info.get('predictions', 'onet_predictions.pickle'))
            self.prediction_file = prediction_file
        with self.prediction_file.open('wb') as out_file:
            pickle.dump(self._predictions, out_file)


class CaffeModelMixin:
    def _infer(self, input_blobs, batch_meta):
        for meta in batch_meta:
            meta['input_shape'] = []
        results = []
        for feed_dict in input_blobs:
            for layer_name, data in feed_dict.items():
                if data.shape != self.inputs[layer_name]:
                    self.net.blobs[layer_name].reshape(*data.shape)
            for meta in batch_meta:
                meta['input_shape'].append(self.inputs)
            results.append(self.net.forward(**feed_dict))

        return results

    @property
    def inputs(self):
        inputs_map = {}
        for input_blob in self.net.inputs:
            inputs_map[input_blob] = self.net.blobs[input_blob].data.shape

        return inputs_map

    def release(self):
        del self.net

    def fit_to_input(self, data, layer_name, layout, precision):
        data_shape = np.shape(data)
        layer_shape = self.inputs[layer_name]
        if len(data_shape) == 5 and len(layer_shape) == 4:
            data = data[0]
            data_shape = np.shape(data)
        data = np.transpose(data, layout) if len(data_shape) == 4 else np.array(data)
        if precision:
            data = data.astype(precision)

        return data

    def automatic_model_search(self, network_info):
        model = Path(network_info.get('model', ''))
        weights = network_info.get('weights')
        if model.is_dir():
            models_list = list(Path(model).glob('{}.prototxt'.format(self.default_model_name)))
            if not models_list:
                models_list = list(Path(model).glob('*.prototxt'))
            if not models_list:
                raise ConfigError('Suitable model description is not detected')
            if len(models_list) != 1:
                raise ConfigError('Several suitable models found, please specify required model')
            model = models_list[0]
            if weights is None or Path(weights).is_dir():
                weights_dir = weights or model.parent
                weights = Path(weights_dir) / model.name.replace('prototxt', 'caffemodel')
                if not weights.exists():
                    weights_list = list(weights_dir.glob('*.caffemodel'))
                    if not weights_list:
                        raise ConfigError('Suitable weights is not detected')
                    if len(weights_list) != 1:
                        raise ConfigError('Several suitable weights found, please specify required explicitly')
                    weights = weights_list[0]
            weights = Path(weights)
            return model, weights


class DLSDKModelMixin:
    def _infer(self, input_blobs, batch_meta):
        for meta in batch_meta:
            meta['input_shape'] = []
        results = []
        for feed_dict in input_blobs:
            input_shapes = {layer_name: data.shape for layer_name, data in feed_dict.items()}
            self._reshape_input(input_shapes)
            results.append(self.exec_network.infer(feed_dict))
            for meta in batch_meta:
                meta['input_shape'].append(self.inputs)

        return results

    def _reshape_input(self, input_shapes):
        del self.exec_network
        self.network.reshape(input_shapes)
        self.exec_network = self.launcher.ie_core.load_network(self.network, self.launcher.device)

    @property
    def inputs(self):
        return self.network.inputs

    def release(self):
        del self.network
        del self.exec_network
        self.launcher.release()

    def fit_to_input(self, data, layer_name, layout, precision):
        layer_shape = tuple(self.inputs[layer_name].shape)
        data_shape = np.shape(data)
        if len(layer_shape) == 4:
            if len(data_shape) == 5:
                data = data[0]
            data = np.transpose(data, layout)
        if precision:
            data = data.astype(precision)

        return data

    def prepare_model(self, launcher):
        launcher_specific_entries = [
            'model', 'weights', 'caffe_model', 'caffe_weights', 'tf_model', 'inputs', 'outputs', '_model_optimizer'
        ]

        def update_mo_params(launcher_config, model_config):
            for entry in launcher_specific_entries:
                if entry not in launcher_config:
                    continue
                if entry in model_config:
                    continue
                model_config[entry] = launcher_config[entry]
            model_mo_flags, model_mo_params = model_config.get('mo_flags', []), model_config.get('mo_params', {})
            launcher_mo_flags, launcher_mo_params = launcher_config.get('mo_flags', []), launcher_config.get(
                'mo_params', {})
            for launcher_flag in launcher_mo_flags:
                if launcher_flag not in model_mo_flags:
                    model_mo_flags.append(launcher_flag)

            for launcher_mo_key, launcher_mo_value in launcher_mo_params.items():
                if launcher_mo_key not in launcher_mo_params:
                    model_mo_params[launcher_mo_key] = launcher_mo_value

            model_config['mo_flags'] = model_mo_flags
            model_config['mo_params'] = model_mo_params

        update_mo_params(launcher.config, self.model_info)
        if 'caffe_model' in self.model_info:
            self.model_info.update(launcher.config)
            model, weights = launcher.convert_model(self.model_info)
        else:
            model, weights = self.auto_model_search(self.model_info)

        return model, weights

    def auto_model_search(self, network_info):
        model = Path(network_info.get('model', ''))
        weights = network_info.get('weights')
        if model.is_dir():
            models_list = list(Path(model).glob('{}.xml'.format(self.default_model_name)))
            if not models_list:
                models_list = list(Path(model).glob('*.xml'))
            if not models_list:
                raise ConfigError('Suitable model description is not detected')
            if len(models_list) != 1:
                raise ConfigError('Several suitable models found, please specify required model')
            model = models_list[0]
            if weights is None or Path(weights).is_dir():
                weights_dir = weights or model.parent
                weights = Path(weights_dir) / model.name.replace('xml', 'bin')
                if not weights.exists():
                    weights_list = list(weights_dir.glob('*.bin'))
                    if not weights_list:
                        raise ConfigError('Suitable weights is not detected')
                    if len(weights_list) != 1:
                        raise ConfigError('Several suitable weights found, please specify required explicitly')
                    weights = weights_list[0]
            weights = Path(weights)
        return model, weights


class CaffeProposalStage(CaffeModelMixin, ProposalBaseStage):
    def __init__(self,  model_info, preprocessor, launcher):
        super().__init__(model_info, preprocessor)
        self.net = launcher.create_network(self.model_info['model'], self.model_info['weights'])
        self.input_feeder = InputFeeder(model_info.get('inputs', []), self.inputs, self.fit_to_input)
        pnet_outs = model_info['outputs']
        pnet_adapter_config = launcher.config.get('adapter', {'type': 'mtcnn_p', **pnet_outs})
        pnet_adapter_config.update({'regions_format': 'hw'})
        self.adapter = create_adapter(pnet_adapter_config)


class CaffeRefineStage(CaffeModelMixin, RefineBaseStage):
    def __init__(self,  model_info, preprocessor, launcher):
        super().__init__(model_info, preprocessor)
        self.net = launcher.create_network(self.model_info['model'], self.model_info['weights'])
        self.input_feeder = InputFeeder(model_info.get('inputs', []), self.inputs,  self.fit_to_input)


class CaffeOutputStage(CaffeModelMixin, OutputBaseStage):
    def __init__(self,  model_info, preprocessor, launcher):
        super().__init__(model_info, preprocessor)
        self.net = launcher.create_network(self.model_info['model'], self.model_info['weights'])
        self.input_feeder = InputFeeder(model_info.get('inputs', []), self.inputs, self.fit_to_input)


class DLSDKProposalStage(DLSDKModelMixin, ProposalBaseStage):
    def __init__(self,  model_info, preprocessor, launcher):
        super().__init__(model_info, preprocessor)
        model_xml, model_bin = self.prepare_model(launcher)
        self.network = launcher.create_ie_network(str(model_xml), str(model_bin))
        self.exec_network = launcher.ie_core.load_network(self.network, launcher.device)
        self.launcher = launcher
        self.input_feeder = InputFeeder(model_info.get('inputs', []), self.inputs, self.fit_to_input)
        pnet_outs = model_info['outputs']
        pnet_adapter_config = launcher.config.get('adapter', {'type': 'mtcnn_p', **pnet_outs})
        # pnet_adapter_config.update({'regions_format': 'hw'})
        self.adapter = create_adapter(pnet_adapter_config)


class DLSDKRefineStage(DLSDKModelMixin, RefineBaseStage):
    def __init__(self,  model_info, preprocessor, launcher):
        super().__init__(model_info, preprocessor)
        model_xml, model_bin = self.prepare_model(launcher)
        self.network = launcher.create_ie_network(str(model_xml), str(model_bin))
        self.exec_network = launcher.ie_core.load_network(self.network, launcher.device)
        self.launcher = launcher
        self.input_feeder = InputFeeder(model_info.get('inputs', []), self.inputs, self.fit_to_input)


class DLSDKOutputStage(DLSDKModelMixin, OutputBaseStage):
    def __init__(self,  model_info, preprocessor, launcher):
        super().__init__(model_info,  preprocessor)
        model_xml, model_bin = self.prepare_model(launcher)
        self.network = launcher.create_ie_network(str(model_xml), str(model_bin))
        self.exec_network = launcher.ie_core.load_network(self.network, launcher.device)
        self.launcher = launcher
        self.input_feeder = InputFeeder(model_info.get('inputs', []), self.inputs, self.fit_to_input)


class MTCNNEvaluator(BaseEvaluator):
    def __init__(
            self, dataset, reader, stages, postprocessing, metrics_executor
    ):
        super().__init__()
        self.dataset = dataset
        self.reader = reader
        self.stages = stages
        self.postprocessing = postprocessing
        self.metrics_executor = metrics_executor
        self._metrics_results = []
        self._annotations, self._predictions = [], []

    def process_dataset(self, stored_predictions, progress_reporter, *args, **kwargs):
        def no_detections(batch_pred):
            return batch_pred[0].size == 0
        if progress_reporter:
            progress_reporter.reset(self.dataset.size)
        for batch_id, (_, batch_annotation) in enumerate(self.dataset):
            batch_identifiers = [annotation.identifier for annotation in batch_annotation]
            batch_input = [self.reader(identifier=identifier) for identifier in batch_identifiers]
            batch_predictions = []
            for stage in self.stages:
                previous_stage_predictions = batch_predictions
                filled_inputs, batch_meta = stage.preprocess_data(copy.deepcopy(batch_input), batch_annotation, previous_stage_predictions)
                batch_predictions = stage.predict(filled_inputs, batch_meta)
                batch_predictions = stage.postprocess_result(
                    batch_identifiers, batch_predictions, batch_meta, previous_stage_predictions
                )
                if no_detections(batch_predictions):
                    break

            batch_annotation, batch_predictions = self.postprocessing.process_batch(batch_annotation, batch_predictions)

            self._annotations.extend(batch_annotation)
            self._predictions.extend(batch_predictions)
            if progress_reporter:
                progress_reporter.update(batch_id, len(batch_predictions))
        for stage in self.stages:
            if stage.store:
                stage.dump_predictions()

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

    def extract_metrics_results(self, print_results=True, ignore_results_formatting=False):
        if not self._metrics_results:
            self.compute_metrics(False, ignore_results_formatting)

        result_presenters = self.metrics_executor.get_metric_presenters()
        extracted_results, extracted_meta = [], []
        for presenter, metric_result in zip(result_presenters, self._metrics_results):
            result, metadata = presenter.extract_result(metric_result)
            if isinstance(result, list):
                extracted_results.extend(result)
                extracted_meta.extend(metadata)
            else:
                extracted_results.append(result)
                extracted_meta.append(metadata)
            if print_results:
                presenter.write_result(metric_result, ignore_results_formatting)

        return extracted_results, extracted_meta

    def print_metrics_results(self, ignore_results_formatting=False):
        if not self._metrics_results:
            self.compute_metrics(True, ignore_results_formatting)
            return
        result_presenters = self.metrics_executor.get_metric_presenters()
        for presenter, metric_result in zip(result_presenters, self._metrics_results):
            presenter.write_result(metric_result, ignore_results_formatting)

    @classmethod
    def from_configs(cls, config):
        dataset_config = config['datasets'][0]
        dataset = Dataset(dataset_config)
        data_reader_config = dataset_config.get('reader', 'opencv_imread')
        data_source = dataset_config['data_source']
        if isinstance(data_reader_config, str):
            data_reader_type = data_reader_config
            data_reader_config = None
        elif isinstance(data_reader_config, dict):
            data_reader_type = data_reader_config['type']
        else:
            raise ConfigError('reader should be dict or string')
        if data_reader_type in REQUIRES_ANNOTATIONS:
            data_source = dataset.annotation
        data_reader = BaseReader.provide(data_reader_type, data_source, data_reader_config)
        models_info = config['network_info']
        launcher_config = config['launchers'][0]
        launcher = create_launcher(launcher_config, delayed_model_loading=True)
        preprocessors_config = dataset_config.get('preprocessing', [])
        stages = build_stages(models_info, preprocessors_config, launcher, config.get('_models'))
        metrics_executor = MetricsExecutor(dataset_config['metrics'], dataset)
        postprocessing = PostprocessingExecutor(dataset_config['postprocessing'])

        return cls(dataset, data_reader, stages, postprocessing, metrics_executor)

    @staticmethod
    def get_processing_info(config):
        module_specific_params = config.get('module_config')
        model_name = config['name']
        dataset_config = module_specific_params['datasets'][0]
        launcher_config = module_specific_params['launchers'][0]
        return (
            model_name, launcher_config['framework'], launcher_config['device'], launcher_config.get('tags'),
            dataset_config['name']
        )

    def release(self):
        for stage in self.stages:
            stage.release()

    def reset(self):
        self.metrics_executor.reset()
        self.dataset.reset()
        for stage in self.stages:
            stage.reset()


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
