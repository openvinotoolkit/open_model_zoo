"""
Copyright (c) 2018-2021 Intel Corporation

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
from collections import defaultdict

import numpy as np

from .metric import PerImageEvaluationMetric
from ..config import StringField, BoolField, ConfigError
from ..representation import (
    SequenceClassificationAnnotation, BERTNamedEntityRecognitionAnnotation, SequenceClassificationPrediction
)


def align_sequences(gt_seq, pred_seq, label_map, convert_token_ids=True, label_mask=None, valid_ids=None):
    aligned_gt, aligned_pred = [], []
    if label_mask is not None and valid_ids is not None:
        gt_seq = np.array(gt_seq)[label_mask]
        pred_seq = np.array(pred_seq[valid_ids])
    start_id = 0 if gt_seq[0] in label_map and label_map[gt_seq[0]] != '[CLS]' else 1
    for gt_tok, pred_tok in zip(gt_seq[start_id:], pred_seq[start_id:]):
        if gt_tok not in label_map:
            continue
        if gt_tok == len(label_map):
            break
        aligned_gt.append(gt_tok if not convert_token_ids else label_map[gt_tok])
        aligned_pred.append(pred_tok if not convert_token_ids else label_map.get(pred_tok, '[unk]'))
    return aligned_gt, aligned_pred


def _prf_divide(numerator, denominator):
    mask = denominator == 0.0
    denominator = denominator.copy()
    denominator[mask] = 1  # avoid infs/nans
    result = numerator / denominator

    if not np.any(mask):
        return result

    # if ``zero_division=1``, set those with denominator == 0 equal to 1
    result[mask] = 0.0

    return result


def _precision_recall_fscore_support(pred_sum, tp_sum, true_sum):
    precision = _prf_divide(
        numerator=tp_sum,
        denominator=pred_sum
    )
    recall = _prf_divide(
        numerator=tp_sum,
        denominator=true_sum,
    )

    denom = precision + recall

    denom[denom == 0] = 1  # avoid division by 0
    f_score = 2 * precision * recall / denom

    return precision, recall, f_score


def extract_tp_actual_correct(y_true, y_pred):
    entities_true = defaultdict(set)
    entities_pred = defaultdict(set)
    for type_name, start, end in get_entities(y_true):
        entities_true[type_name].add((start, end))
    for type_name, start, end in get_entities(y_pred):
        entities_pred[type_name].add((start, end))

    target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))

    tp_sum = defaultdict(lambda: 0)
    pred_sum = defaultdict(lambda: 0)
    true_sum = defaultdict(lambda: 0)
    for type_name in target_names:
        entities_true_type = entities_true.get(type_name, set())
        entities_pred_type = entities_pred.get(type_name, set())
        tp_sum[type_name] += len(entities_true_type & entities_pred_type)
        pred_sum[type_name] += len(entities_pred_type)
        true_sum[type_name] += len(entities_true_type)

    return pred_sum, tp_sum, true_sum, target_names


def get_entities(seq, suffix=False):
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        if suffix:
            tag = chunk[-1]
            type_ = chunk[:-1].rsplit('-', maxsplit=1)[0] or '_'
        else:
            tag = chunk[0]
            type_ = chunk[1:].split('-', maxsplit=1)[-1] or '_'

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    chunk_end = False

    if prev_tag == 'E':
        chunk_end = True
    if prev_tag == 'S':
        chunk_end = True

    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'S':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    chunk_start = False

    if tag == 'B':
        chunk_start = True
    if tag == 'S':
        chunk_start = True

    if prev_tag == 'E' and tag == 'E':
        chunk_start = True
    if prev_tag == 'E' and tag == 'I':
        chunk_start = True
    if prev_tag == 'S' and tag == 'E':
        chunk_start = True
    if prev_tag == 'S' and tag == 'I':
        chunk_start = True
    if prev_tag == 'O' and tag == 'E':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


class NERAccuracy(PerImageEvaluationMetric):
    __provider__ = 'ner_accuracy'
    annotation_types = (SequenceClassificationAnnotation, BERTNamedEntityRecognitionAnnotation,)
    prediction_types = (SequenceClassificationPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'label_map': StringField(optional=True, default='label_map', description="Label map."),
            'include_all_tokens': BoolField(
                optional=True, default=False,
                description='should all tokens will be considered during metirc calculation or not'
            )
        })
        return parameters

    def configure(self):
        label_map = self.get_value_from_config('label_map')
        if self.dataset.metadata:
            self.labels = self.dataset.metadata.get(label_map)
            if not self.labels:
                raise ConfigError('ner_accuracy metric requires label_map providing in dataset_meta'
                                  'Please provide dataset meta file or regenerate annotation')
        else:
            raise ConfigError('ner_accuracy metric requires dataset metadata'
                              'Please provide dataset meta file or regenerate annotation')
        self.include_all_tokens = self.get_value_from_config('include_all_tokens')
        self.correct = 0
        self.total = 0

    def update(self, annotation, prediction):
        gt_seq = annotation.label
        pred_seq = prediction.label
        label_mask = annotation.label_mask if not self.include_all_tokens else None
        valid_ids = annotation.valid_ids if not self.include_all_tokens else None
        y_true, y_pred = align_sequences(
            gt_seq, pred_seq, self.labels, False, label_mask, valid_ids
        )
        nb_correct = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
        nb_true = len(y_true)
        self.correct += nb_correct
        self.total += nb_true

        return nb_correct / nb_true if nb_true else 0

    def evaluate(self, annotations, predictions):
        return self.correct / self.total if self.total else 0

    def reset(self):
        self.correct = 0
        self.total = 0


class NERPrecision(PerImageEvaluationMetric):
    __provider__ = 'ner_recall'
    annotation_types = (SequenceClassificationAnnotation, BERTNamedEntityRecognitionAnnotation,)
    prediction_types = (SequenceClassificationPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'label_map': StringField(optional=True, default='label_map', description="Label map."),
            'include_all_tokens': BoolField(
                optional=True, default=False,
                description='should all tokens will be considered during metirc calculation or not'
            )})
        return parameters

    def configure(self):
        label_map = self.get_value_from_config('label_map')
        if self.dataset.metadata:
            self.labels = self.dataset.metadata.get(label_map)
            if not self.labels:
                raise ConfigError('ner_recall metric requires label_map providing in dataset_meta'
                                  'Please provide dataset meta file or regenerate annotation')
        else:
            raise ConfigError('ner_recall metric requires dataset metadata'
                              'Please provide dataset meta file or regenerate annotation')
        self.reset()
        self.imclude_all_tokens = self.get_value_from_config('include_all_tokens')

    def update(self, annotation, prediction):
        gt_seq = annotation.label
        pred_seq = prediction.label
        label_mask = annotation.label_mask if not self.include_all_tokens else None
        valid_ids = annotation.valid_ids if not self.include_all_tokens else None
        y_true, y_pred = align_sequences(gt_seq, pred_seq, self.labels, False, label_mask, valid_ids)
        pred_sum, tp_sum, true_sum, target_names = extract_tp_actual_correct(y_true, y_pred)
        for type_name in target_names:

            self.tp_sum[type_name] += tp_sum[type_name]
            self.pred_sum[type_name] += pred_sum[type_name]
            self.true_sum[type_name] += true_sum[type_name]
        pred_sum_arr = np.array(list(pred_sum.values()))
        tp_sum_arr = np.array(list(tp_sum.values()))
        true_sum_arr = np.array(list(true_sum.values()))
        _, r, _ = _precision_recall_fscore_support(pred_sum_arr, tp_sum_arr, true_sum_arr)
        return r

    def evaluate(self, annotations, predictions):
        pred_sum_arr = np.array(list(self.pred_sum.values()))
        tp_sum_arr = np.array(list(self.tp_sum.values()))
        true_sum_arr = np.array(list(self.true_sum.values()))
        _, recall, _ = _precision_recall_fscore_support(pred_sum_arr, tp_sum_arr, true_sum_arr)
        self.meta['names'] = list(self.pred_sum.keys())

        return recall

    def reset(self):
        self.tp_sum = defaultdict(lambda: 0)
        self.pred_sum = defaultdict(lambda: 0)
        self.true_sum = defaultdict(lambda: 0)


class NERRecall(PerImageEvaluationMetric):
    __provider__ = 'ner_precision'
    annotation_types = (SequenceClassificationAnnotation, BERTNamedEntityRecognitionAnnotation,)
    prediction_types = (SequenceClassificationPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'label_map': StringField(optional=True, default='label_map', description="Label map."),
            'include_all_tokens': BoolField(
                optional=True, default=False,
                description='should all tokens will be considered during metric calculation or not'
            )
        })
        return parameters

    def configure(self):
        label_map = self.get_value_from_config('label_map')
        if self.dataset.metadata:
            self.labels = self.dataset.metadata.get(label_map)
            if not self.labels:
                raise ConfigError('ner_precision metric requires label_map providing in dataset_meta'
                                  'Please provide dataset meta file or regenerate annotation')
        else:
            raise ConfigError('ner_precision metric requires dataset metadata'
                              'Please provide dataset meta file or regenerate annotation')
        self.reset()
        self.include_all_tokens = self.get_value_from_config('include_all_tokens')

    def update(self, annotation, prediction):
        gt_seq = annotation.label
        pred_seq = prediction.label
        label_mask = annotation.label_mask if not self.include_all_tokens else None
        valid_ids = annotation.valid_ids if not self.include_all_tokens else None
        y_true, y_pred = align_sequences(gt_seq, pred_seq, self.labels, False, label_mask, valid_ids)
        pred_sum, tp_sum, true_sum, target_names = extract_tp_actual_correct(y_true, y_pred)
        for type_name in target_names:
            self.tp_sum[type_name] += tp_sum[type_name]
            self.pred_sum[type_name] += pred_sum[type_name]
            self.true_sum[type_name] += true_sum[type_name]
        pred_sum_arr = np.array(list(pred_sum.values()))
        tp_sum_arr = np.array(list(tp_sum.values()))
        true_sum_arr = np.array(list(true_sum.values()))
        p, _, _ = _precision_recall_fscore_support(pred_sum_arr, tp_sum_arr, true_sum_arr)
        return p

    def evaluate(self, annotations, predictions):
        pred_sum_arr = np.array(list(self.pred_sum.values()))
        tp_sum_arr = np.array(list(self.tp_sum.values()))
        true_sum_arr = np.array(list(self.true_sum.values()))
        precision, _, _ = _precision_recall_fscore_support(pred_sum_arr, tp_sum_arr, true_sum_arr)
        self.meta['names'] = list(self.pred_sum.keys())

        return precision

    def reset(self):
        self.tp_sum = defaultdict(lambda: 0)
        self.pred_sum = defaultdict(lambda: 0)
        self.true_sum = defaultdict(lambda: 0)


class NERFScore(PerImageEvaluationMetric):
    __provider__ = 'ner_f_score'
    annotation_types = (SequenceClassificationAnnotation, BERTNamedEntityRecognitionAnnotation,)
    prediction_types = (SequenceClassificationPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'label_map': StringField(optional=True, default='label_map', description="Label map."),
            'include_all_tokens': BoolField(
                optional=True, default=False,
                description='should all tokens will be considered during metirc calculation or not'
            )
        })
        return parameters

    def configure(self):
        label_map = self.get_value_from_config('label_map')
        if self.dataset.metadata:
            self.labels = self.dataset.metadata.get(label_map)
            if not self.labels:
                raise ConfigError('ner_f_score metric requires label_map providing in dataset_meta'
                                  'Please provide dataset meta file or regenerate annotation')
        else:
            raise ConfigError('ner_f_score metric requires dataset metadata'
                              'Please provide dataset meta file or regenerate annotation')
        self.reset()
        self.include_all_tokens = self.get_value_from_config('include_all_tokens')

    def update(self, annotation, prediction):
        gt_seq = annotation.label
        pred_seq = prediction.label
        label_mask = annotation.label_mask if not self.include_all_tokens else None
        valid_ids = annotation.valid_ids if not self.include_all_tokens else None
        y_true, y_pred = align_sequences(gt_seq, pred_seq, self.labels, True, label_mask, valid_ids)
        pred_sum, tp_sum, true_sum, target_names = extract_tp_actual_correct(y_true, y_pred)
        for type_name in target_names:
            self.tp_sum[type_name] += tp_sum[type_name]
            self.pred_sum[type_name] += pred_sum[type_name]
            self.true_sum[type_name] += true_sum[type_name]
        pred_sum_arr = np.array(list(pred_sum.values()))
        tp_sum_arr = np.array(list(tp_sum.values()))
        true_sum_arr = np.array(list(true_sum.values()))
        _, _, f = _precision_recall_fscore_support(pred_sum_arr, tp_sum_arr, true_sum_arr)
        return f

    def evaluate(self, annotations, predictions):
        pred_sum_arr = np.array(list(self.pred_sum.values()))
        tp_sum_arr = np.array(list(self.tp_sum.values()))
        true_sum_arr = np.array(list(self.true_sum.values()))
        _, _, f_score = _precision_recall_fscore_support(pred_sum_arr, tp_sum_arr, true_sum_arr)
        self.meta['names'] = list(self.pred_sum.keys())

        return f_score

    def reset(self):
        self.tp_sum = defaultdict(lambda: 0)
        self.pred_sum = defaultdict(lambda: 0)
        self.true_sum = defaultdict(lambda: 0)
