"""
 Copyright (c) 2018 Intel Corporation

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
from collections import defaultdict, namedtuple
from sklearn.metrics import auc, precision_recall_curve
from sklearn.metrics.base import _average_binary_score
import numpy as np

from ..representation import (ReIdentificationClassificationAnnotation, ReIdentificationAnnotation,
                              ReIdentificationPrediction)
from ..config import BaseField, BoolField, NumberField
from .metric import BaseMetricConfig, FullDatasetEvaluationMetric

PairDesc = namedtuple('PairDesc', 'image1 image2 same')


class CMCScore(FullDatasetEvaluationMetric):
    """Cumulative Matching Characteristics (CMC) score

    Config:
        annotation: reid annotation
        prediction: predicted embeddings
        topk: number of k highest ranked samples to consider when matching
        separate_camera_set: should identities from the same camera view be filtered out
        single_gallery_shot: each identity has only one instance in the gallery
        number_single_shot_repeats: number of repeats for single_gallery_shot setting (required for CUHK)
        first_match_break: break on first matched gallery sample
    """
    __provider__ = 'cmc'
    annotation_types = (ReIdentificationAnnotation, )
    prediction_types = (ReIdentificationPrediction, )

    def validate_config(self):
        class _CMCConfigValidator(BaseMetricConfig):
            top_k = NumberField(floats=False, min_value=1, optional=True)
            separate_camera_set = BoolField(optional=True)
            single_gallery_shot = BoolField(optional=True)
            first_match_break = BoolField(optional=True)
            number_single_shot_repeats = NumberField(floats=False, optional=True)

        validator = _CMCConfigValidator('cmc', on_extra_argument=_CMCConfigValidator.ERROR_ON_EXTRA_ARGUMENT)
        validator.validate(self.config)

    def configure(self):
        self.top_k = self.config.get('top_k', 1)
        self.separate_camera_set = self.config.get('separate_camera_set', False)
        self.single_gallery_shot = self.config.get('single_gallery_shot', False)
        self.first_match_break = self.config.get('first_match_break', True)
        self.number_single_shot_repeats = self.config.get('number_single_shot_repeats', 10)

    def evaluate(self, annotations, predictions):
        dist_matrix = _distance_matrix(annotations, predictions)
        gallery_cameras, gallery_pids, query_cameras, quey_pids = _get_gallery_query_pids(annotations)

        _cmc_score = _eval_cmc(dist_matrix,
                               quey_pids, gallery_pids, query_cameras, gallery_cameras, self.separate_camera_set,
                               self.single_gallery_shot, self.first_match_break, self.number_single_shot_repeats)

        return _cmc_score[self.top_k - 1]


class ReidMAP(FullDatasetEvaluationMetric):
    """Mean Average Precision score

    Config:
        annotation: reid annotation
        prediction: predicted embeddings
        uninterpolated_auc: should area under precision recall curve be computed using trapezoidal rule or directly
    """
    __provider__ = 'reid_map'
    annotation_types = (ReIdentificationAnnotation, )
    prediction_types = (ReIdentificationPrediction, )

    def validate_config(self):
        class _ReidMapConfig(BaseMetricConfig):
            interpolated_auc = BoolField(optional=True)

        validator = _ReidMapConfig('reid_map', on_extra_argument=_ReidMapConfig.ERROR_ON_EXTRA_ARGUMENT)
        validator.validate(self.config)

    def configure(self):
        self.interpolated_auc = self.config.get('interpolated_auc', True)

    def evaluate(self, annotations, predictions):
        dist_matrix = _distance_matrix(annotations, predictions)
        gallery_cameras, gallery_pids, query_cameras, quey_pids = _get_gallery_query_pids(annotations)

        _map_score = _eval_map(dist_matrix,
                               quey_pids, gallery_pids, query_cameras, gallery_cameras, self.interpolated_auc)
        return _map_score


class PairwiseAccuracy(FullDatasetEvaluationMetric):
    __provider__ = 'pairwise_accuracy'
    annotation_types = (ReIdentificationClassificationAnnotation, )
    prediction_types = (ReIdentificationPrediction, )

    def validate_config(self):
        class _PWAccConfig(BaseMetricConfig):
            min_score = BaseField(optional=True)

        validator = _PWAccConfig('pairwise_accuracy', on_extra_argument=_PWAccConfig.ERROR_ON_EXTRA_ARGUMENT)
        validator.validate(self.config)

    def configure(self):
        self.min_score = self.config.get('min_score', 'train_median')

    def evaluate(self, annotations, predictions):
        embd_distances, pairs = _get_embedding_distances(annotations, predictions)
        min_score = self.min_score

        if min_score == 'train_median':
            train_distances, _train_pairs = _get_embedding_distances(annotations, predictions, train=True)
            min_score = np.median(train_distances)

        embd_same_class = embd_distances < min_score

        accuracy = 0
        for i, pair in enumerate(pairs):
            same_label = pair.same
            out_same = embd_same_class[i]

            correct_prediction = same_label and out_same or (not same_label and not out_same)

            if correct_prediction:
                accuracy += 1
        return float(accuracy) / len(pairs)


class PairwiseAccuracySubsets(FullDatasetEvaluationMetric):
    __provider__ = 'pairwise_accuracy_subsets'
    annotation_types = (ReIdentificationClassificationAnnotation, )
    prediction_types = (ReIdentificationPrediction, )

    def validate_config(self):
        class _PWAccConfig(BaseMetricConfig):
            subset_number = NumberField(optional=True, min_value=1, floats=False)
        validator = _PWAccConfig('pairwise_accuracy', on_extra_argument=_PWAccConfig.ERROR_ON_EXTRA_ARGUMENT)
        validator.validate(self.config)

    def configure(self):
        self.subset_num = self.config.get('subset_number', 10)
        self.accuracy_metric = PairwiseAccuracy(self.config, self.dataset)

    def evaluate(self, annotations, predictions):
        subset_results = []
        first_images_annotations = list(filter(
            lambda annotation: (len(annotation.negative_pairs) > 0 or len(annotation.positive_pairs) > 0), annotations))
        idx_subsets = self.make_subsets(self.subset_num, len(first_images_annotations))
        for i in range(self.subset_num):
            test_subset = self.get_subset(first_images_annotations, idx_subsets[i]['test'])
            test_subset = self.mark_subset(test_subset, False)
            train_subset = self.get_subset(first_images_annotations, idx_subsets[i]['train'])
            train_subset = self.mark_subset(train_subset)
            subset_result = self.accuracy_metric.evaluate(test_subset+train_subset, predictions)
            subset_results.append(subset_result)
        return np.mean(subset_results)

    @staticmethod
    def make_subsets(subset_num, dataset_size):
        subsets = []
        if subset_num > dataset_size:
            raise ValueError('It is impossible to divide dataset on more than number of annotations subsets.')
        for i in range(subset_num):
            lower_bnd = i * dataset_size // subset_num
            upper_bnd = (i + 1) * dataset_size // subset_num
            subset_test = [(lower_bnd, upper_bnd)]
            subset_train = [(0, lower_bnd), (upper_bnd, dataset_size)]
            subsets.append({'test': subset_test, 'train': subset_train})
        return subsets

    @staticmethod
    def mark_subset(subset_annotations, train=True):
        for annotation in subset_annotations:
            annotation.metadata['train'] = train
        return subset_annotations

    @staticmethod
    def get_subset(container, subset_bounds):
        subset = []
        for bound in subset_bounds:
            subset += container[bound[0]: bound[1]]
        return subset


def _extract_embeddings(annotation, prediction, query):
    return np.stack([pred.embedding for pred, ann in zip(prediction, annotation) if ann.query == query])


def _get_gallery_query_pids(annotation):
    gallery_pids = np.asarray([ann.person_id for ann in annotation if not ann.query])
    quey_pids = np.asarray([ann.person_id for ann in annotation if ann.query])
    gallery_cameras = np.asarray([ann.camera_id for ann in annotation if not ann.query])
    query_cameras = np.asarray([ann.camera_id for ann in annotation if ann.query])
    return gallery_cameras, gallery_pids, query_cameras, quey_pids


def _distance_matrix(annotation, prediction):
    gallery_embeddings = _extract_embeddings(annotation, prediction, query=False)
    query_embeddings = _extract_embeddings(annotation, prediction, query=True)

    dist = 1. - np.matmul(gallery_embeddings, np.transpose(query_embeddings)).T
    return dist


def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask


def _eval_map(distmat, query_ids, gallery_ids, query_cams, gallery_cams, interpolated_auc=False):
    number_queries, _number_gallery = distmat.shape
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute AP for each query
    aps = []
    for i in range(number_queries):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))

        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true):
            continue

        aps.append(binary_average_precision(y_true, y_score, interpolated_auc=interpolated_auc))

    if not aps:
        raise RuntimeError("No valid query")
    return np.mean(aps)


def _eval_cmc(distmat, query_ids, gallery_ids,
              query_cams, gallery_cams,
              separate_camera_set=False,
              single_gallery_shot=False,
              first_match_break=False,
              number_single_shot_repeats=10, topk=100):
    number_queries, _number_gallery = distmat.shape

    if not single_gallery_shot:
        number_single_shot_repeats = 1
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])  # type: np.ndarray

    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = 0
    for i in range(number_queries):

        valid = _get_valid_subset(gallery_cams, gallery_ids, i, indices, query_cams, query_ids, separate_camera_set)

        if not np.any(matches[i, valid]):
            continue

        if single_gallery_shot:
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)

        for _ in range(number_single_shot_repeats):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                # required for correct validation on CUHK datasets
                # http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html
                sampled = (valid & _unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]
            delta = 1. / (len(index) * number_single_shot_repeats)
            for j, k in enumerate(index):
                if k - j >= topk:
                    break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta

        num_valid_queries += 1

    if num_valid_queries == 0:
        raise RuntimeError("No valid query")

    return ret.cumsum() / num_valid_queries


def _get_valid_subset(gallery_cams, gallery_ids, query_index, indices, query_cams, query_ids, separate_camera_set):
    # Filter out the same id and same camera
    valid = ((gallery_ids[indices[query_index]] != query_ids[query_index]) |
             (gallery_cams[indices[query_index]] != query_cams[query_index]))
    if separate_camera_set:
        # Filter out samples from same camera
        valid &= (gallery_cams[indices[query_index]] != query_cams[query_index])
    return valid


def _get_embedding_distances(annotation, prediction, train=False):
    image_indexes = {}
    for i, pred in enumerate(prediction):
        image_indexes[pred.identifier] = i
    # get pairs
    pairs = []
    for image1 in annotation:
        if train != image1.metadata.get("train", False):
            continue

        for image2 in image1.positive_pairs:
            pairs.append(PairDesc(image_indexes[image1.identifier], image_indexes[image2], True))
        for image2 in image1.negative_pairs:
            pairs.append(PairDesc(image_indexes[image1.identifier], image_indexes[image2], False))
    embd1 = np.asarray([prediction[idx].embedding for idx, _, _ in pairs])
    embd2 = np.asarray([prediction[idx].embedding for _, idx, _ in pairs])

    embd_distances = 0.5 * (1 - np.sum(embd1 * embd2, axis=1))
    return embd_distances, pairs

def binary_average_precision(y_true, y_score, interpolated_auc=True):
    def _average_precision(y_true, y_score, sample_weight=None):
        precision, recall, _ = precision_recall_curve(y_true, y_score)

        if not interpolated_auc:
            # Return the step function integral
            # The following works because the last entry of precision is
            # guaranteed to be 1, as returned by precision_recall_curve
            return -1 * np.sum(np.diff(recall) * np.array(precision)[:-1])
        return auc(recall, precision)

    return _average_binary_score(_average_precision,
                                 y_true, y_score, average="macro")
