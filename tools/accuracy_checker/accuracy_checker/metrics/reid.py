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

import warnings
from collections import defaultdict, namedtuple
import numpy as np

from ..representation import (
    ReIdentificationClassificationAnnotation,
    ReIdentificationAnnotation,
    PlaceRecognitionAnnotation,
    ReIdentificationPrediction
)
from ..config import BaseField, BoolField, NumberField, StringField
from .metric import FullDatasetEvaluationMetric


def _auc(x, y):
    if x.shape[0] < 2:
        raise ValueError('At least 2 points are needed to compute'
                         ' area under curve, but x.shape = {}'.format(x.shape))
    direction = 1
    dx = np.diff(x)
    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1
        else:
            raise ValueError("x is neither increasing nor decreasing "
                             ": {}.".format(x))
    area = direction * np.trapz(y, x)
    return area


def _binary_clf_curve(y_true, y_score):
    pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    weight = 1.

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = np.cumsum((y_true * weight), axis=None, dtype=np.float64)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]


def _precision_recall_curve(y_true, probas_pred):

    fps, tps, thresholds = _binary_clf_curve(y_true, probas_pred,)

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


try:
    from sklearn.metrics import auc, precision_recall_curve
except ImportError:
    auc = _auc
    precision_recall_curve = _precision_recall_curve


PairDesc = namedtuple('PairDesc', 'image1 image2 same')


def _average_binary_score(binary_metric, y_true, y_score):
    def binary_target(y):
        return not (len(np.unique(y)) > 2) or (y.ndim >= 2 and len(y[0]) > 1)

    if binary_target(y_true):
        return binary_metric(y_true, y_score)

    y_true = y_true.ravel()
    y_score = y_score.ravel()

    n_classes = y_score.shape[1]
    score = np.zeros((n_classes,))
    for c in range(n_classes):
        y_true_c = y_true.take([c], axis=1).ravel()
        y_score_c = y_score.take([c], axis=1).ravel()
        score[c] = binary_metric(y_true_c, y_score_c)

    return score


class CMCScore(FullDatasetEvaluationMetric):
    """
    Cumulative Matching Characteristics (CMC) score.

    Config:
        annotation: reid annotation.
        prediction: predicted embeddings.
        top_k: number of k highest ranked samples to consider when matching.
        separate_camera_set: should identities from the same camera view be filtered out.
        single_gallery_shot: each identity has only one instance in the gallery.
        number_single_shot_repeats: number of repeats for single_gallery_shot setting.
        first_match_break: break on first matched gallery sample.
    """

    __provider__ = 'cmc'

    annotation_types = (ReIdentificationAnnotation, )
    prediction_types = (ReIdentificationPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'top_k': NumberField(
                value_type=int, min_value=1, default=1, optional=True,
                description="Number of k highest ranked samples to consider when matching."
            ),
            'separate_camera_set': BoolField(
                optional=True, default=False, description="Should identities from the same camera view be filtered out."
            ),
            'single_gallery_shot': BoolField(
                optional=True, default=False, description="Each identity has only one instance in the gallery."
            ),
            'first_match_break': BoolField(
                optional=True, default=True, description="Break on first matched gallery sample."
            ),
            'number_single_shot_repeats': NumberField(
                value_type=int, optional=True, default=10,
                description="Number of repeats for single_gallery_shot setting (required for CUHK)."
            )
        })
        return parameters

    def configure(self):
        self.top_k = self.get_value_from_config('top_k')
        self.separate_camera_set = self.get_value_from_config('separate_camera_set')
        self.single_gallery_shot = self.get_value_from_config('single_gallery_shot')
        self.first_match_break = self.get_value_from_config('first_match_break')
        self.number_single_shot_repeats = self.get_value_from_config('number_single_shot_repeats')

    def evaluate(self, annotations, predictions):
        dist_matrix = distance_matrix(annotations, predictions)
        if np.size(dist_matrix) == 0:
            warnings.warn('Gallery and query ids are not matched. CMC score can not be calculated.')
            return 0
        gallery_cameras, gallery_pids, query_cameras, query_pids = get_gallery_query_pids(annotations)

        _cmc_score = eval_cmc(
            dist_matrix, query_pids, gallery_pids, query_cameras, gallery_cameras, self.separate_camera_set,
            self.single_gallery_shot, self.first_match_break, self.number_single_shot_repeats
        )

        return _cmc_score[self.top_k - 1]


class ReidMAP(FullDatasetEvaluationMetric):
    """
    Mean Average Precision score.

    Config:
        annotation: reid annotation.
        prediction: predicted embeddings.
        interpolated_auc: should area under precision recall curve be computed using trapezoidal rule or directly.
    """

    __provider__ = 'reid_map'

    annotation_types = (ReIdentificationAnnotation, )
    prediction_types = (ReIdentificationPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'interpolated_auc': BoolField(
                optional=True, default=True, description="Should area under precision recall"
                                                         " curve be computed using trapezoidal rule or directly."
            )
        })
        return parameters

    def configure(self):
        self.interpolated_auc = self.get_value_from_config('interpolated_auc')

    def evaluate(self, annotations, predictions):
        dist_matrix = distance_matrix(annotations, predictions)
        if np.size(dist_matrix) == 0:
            warnings.warn('Gallery and query ids are not matched. ReID mAP can not be calculated.')
            return 0
        gallery_cameras, gallery_pids, query_cameras, query_pids = get_gallery_query_pids(annotations)

        return eval_map(
            dist_matrix, query_pids, gallery_pids, query_cameras, gallery_cameras, self.interpolated_auc
        )


class PairwiseAccuracy(FullDatasetEvaluationMetric):
    __provider__ = 'pairwise_accuracy'

    annotation_types = (ReIdentificationClassificationAnnotation, )
    prediction_types = (ReIdentificationPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'min_score': BaseField(
                optional=True, default='train_median',
                description="Min score for determining that objects are different. "
                            "You can provide value or use train_median or best_train_threshold values "
                            "which will be calculated if annotations has training subset."
            ),
            'distance_method': StringField(
                optional=True, default='euclidian_distance',
                description='Allows to choose one of the distance calculation methods',
                choices=['euclidian_distance', 'cosine_distance']
            ),
            'subtract_mean': BoolField(
                optional=True, default=False, description='Allows to subtract mean calculated on train embeddings '
                                                          'before calculating the distance'
            )
        })
        return parameters

    def configure(self):
        self.min_score = self.get_value_from_config('min_score')
        self.distance_method = self.get_value_from_config('distance_method')
        self.subtract_mean = self.get_value_from_config('subtract_mean')

    def evaluate(self, annotations, predictions):
        min_score = self.min_score
        if min_score == 'train_median':
            train_distances, _train_pairs, mean = get_embedding_distances(annotations, predictions, train=True,
                                                                          distance_method=self.distance_method,
                                                                          save_mean=self.subtract_mean)
            min_score_value = np.median(train_distances)
        elif min_score == 'best_train_threshold':
            train_distances, train_pairs, mean = get_embedding_distances(annotations, predictions, train=True,
                                                                         distance_method=self.distance_method,
                                                                         save_mean=self.subtract_mean)
            thresholds = np.arange(0, 4, 0.01)
            accuracy_train = np.zeros((thresholds.size))
            for threshold_idx, threshold in enumerate(thresholds):
                train_same_class = train_distances < threshold
                accuracy = 0
                for i, pair in enumerate(train_pairs):
                    same_label = pair.same
                    out_same = train_same_class[i]

                    correct_prediction = (same_label and out_same) or (not same_label and not out_same)

                    if correct_prediction:
                        accuracy += 1
                accuracy_train[threshold_idx] = accuracy
            min_score_value = thresholds[np.argmax(accuracy_train)]
        else:
            min_score_value = min_score
            mean = 0.0

        embed_distances, pairs, _ = get_embedding_distances(annotations, predictions, mean=mean,
                                                            distance_method=self.distance_method)
        if not pairs:
            return np.nan

        embed_same_class = embed_distances < min_score_value

        accuracy = 0
        for i, pair in enumerate(pairs):
            same_label = pair.same
            out_same = embed_same_class[i]

            correct_prediction = (same_label and out_same) or (not same_label and not out_same)

            if correct_prediction:
                accuracy += 1

        return float(accuracy) / len(pairs)


class PairwiseAccuracySubsets(FullDatasetEvaluationMetric):
    __provider__ = 'pairwise_accuracy_subsets'

    annotation_types = (ReIdentificationClassificationAnnotation, )
    prediction_types = (ReIdentificationPrediction, )

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'subset_number': NumberField(
                optional=True, min_value=1, value_type=int, default=10, description="Number of subsets for separating."
            ),
            'min_score': BaseField(
                optional=True, default='train_median',
                description="Min score for determining that objects are different. "
                            "You can provide value or use train_median or best_train_threshold values "
                            "which will be calculated if annotations has training subset."
            ),
            'distance_method': StringField(
                optional=True, default='euclidian_distance',
                description='Allows to choose one of the distance calculation methods',
                choices=['euclidian_distance', 'cosine_distance']
            ),
            'subtract_mean': BoolField(
                optional=True, default=False, description='Allows to subtract mean calculated on train embeddings '
                                                          'before calculating the distance'
            )
        })
        return params

    def configure(self):
        self.subset_num = self.get_value_from_config('subset_number')
        config_copy = self.config.copy()
        if 'subset_number' in config_copy:
            config_copy.pop('subset_number')
        self.accuracy_metric = PairwiseAccuracy(config_copy, self.dataset)

    def evaluate(self, annotations, predictions):
        subset_results = []
        first_images_annotations = list(filter(
            lambda annotation: (len(annotation.negative_pairs) > 0 or len(annotation.positive_pairs) > 0), annotations
        ))

        idx_subsets = self.make_subsets(self.subset_num, len(first_images_annotations))
        if not idx_subsets:
            return 0

        for subset in range(self.subset_num):
            test_subset = self.get_subset(first_images_annotations, idx_subsets[subset]['test'])
            test_subset = self.mark_subset(test_subset, False)

            train_subset = self.get_subset(first_images_annotations, idx_subsets[subset]['train'])
            train_subset = self.mark_subset(train_subset)

            subset_result = self.accuracy_metric.evaluate(test_subset+train_subset, predictions)
            if not np.isnan(subset_result):
                subset_results.append(subset_result)

        return np.mean(subset_results) if subset_results else 0

    @staticmethod
    def make_subsets(subset_num, dataset_size):
        subsets = []
        if subset_num > dataset_size:
            warnings.warn('It is impossible to divide dataset on more than number of annotations subsets.')
            return []

        for subset in range(subset_num):
            lower_bnd = subset * dataset_size // subset_num
            upper_bnd = (subset + 1) * dataset_size // subset_num
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


class FaceRecognitionTAFAPairMetric(FullDatasetEvaluationMetric):
    __provider__ = 'face_recognition_tafa_pair_metric'

    annotation_types = (ReIdentificationAnnotation, )
    prediction_types = (ReIdentificationPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'threshold': NumberField(
                value_type=float,
                min_value=0,
                optional=False,
                description='Threshold value to identify pair of faces as matched'
            )
        })
        return parameters

    def configure(self):
        self.threshold = self.get_value_from_config('threshold')

    def submit_all(self, annotations, predictions):
        return self.evaluate(annotations, predictions)

    def evaluate(self, annotations, predictions):
        tp = fp = tn = fn = 0
        pairs = regroup_pairs(annotations, predictions)

        for pair in pairs:
            # Dot product of embeddings
            prediction = np.dot(predictions[pair.image1].embedding, predictions[pair.image2].embedding)

            # Similarity scale-shift
            prediction = (prediction + 1) / 2

            # Calculate metrics
            if pair.same: # Pairs that match
                if prediction > self.threshold:
                    tp += 1
                else:
                    fp += 1
            else:
                if prediction < self.threshold:
                    tn += 1
                else:
                    fn += 1

        return [(tp+tn) / (tp+fp+tn+fn)]


class NormalizedEmbeddingAccuracy(FullDatasetEvaluationMetric):
    """
    Accuracy score calculated with normalized embedding dot products
    """
    __provider__ = 'normalized_embedding_accuracy'

    annotation_types = (ReIdentificationAnnotation, )
    prediction_types = (ReIdentificationPrediction, )

    def configure(self):
        self.top_k = self.get_value_from_config('top_k')

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'top_k': NumberField(
                value_type=int, min_value=1, default=1, optional=True,
                description="Number of k highest ranked samples to consider when matching."
            )
        })
        return parameters

    @staticmethod
    def extract_person_id(annotations, query=False):
        return np.array([a.person_id for a in annotations if a.query == query])

    @staticmethod
    def extract_cam_id(annotations, query=False):
        return np.array([a.camera_id for a in annotations if a.query == query])

    @staticmethod
    def eval_valid_matrix(gallery_person_ids, gallery_cam_ids, query_person_ids, query_cam_ids):
        person_id_mask = np.tile(gallery_person_ids, (len(query_person_ids), 1))
        person_id_mask = (person_id_mask == np.tile(query_person_ids,
                                                    (len(gallery_person_ids), 1)).T)
        cam_id_mask = np.tile(gallery_cam_ids, (len(query_cam_ids), 1))
        cam_id_mask = (cam_id_mask == np.tile(query_cam_ids, (len(gallery_cam_ids), 1)).T)
        return 1 - person_id_mask*cam_id_mask


    def evaluate(self, annotations, predictions):
        gallery_embeddings = extract_embeddings(annotations, predictions, query=False)
        gallery_person_ids = self.extract_person_id(annotations, False)
        gallery_cam_ids = self.extract_cam_id(annotations, False)
        query_embeddings = extract_embeddings(annotations, predictions, query=True)
        query_person_ids = self.extract_person_id(annotations, True)
        query_cam_ids = self.extract_cam_id(annotations, True)

        valid_mask = self.eval_valid_matrix(gallery_person_ids, gallery_cam_ids, query_person_ids, query_cam_ids)
        if np.size(gallery_embeddings) == 0 or np.size(query_embeddings) == 0:
            return 0

        gallery_embeddings = gallery_embeddings / np.linalg.norm(gallery_embeddings, axis=1).reshape(-1, 1)
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1).reshape(-1, 1)
        dist_mat = np.matmul(query_embeddings, gallery_embeddings.transpose())
        dist_mat *= valid_mask
        sorted_idx = np.argsort(-dist_mat, axis=1)[:, :self.top_k]

        pred_top_k_query_ids = gallery_person_ids[sorted_idx].T
        query_person_ids = np.tile(query_person_ids, (self.top_k, 1))

        tp = np.any(query_person_ids == pred_top_k_query_ids, axis=0).sum()
        fp = query_person_ids.shape[1] - tp

        if (tp+fp) == 0:
            return 0
        return tp/(tp+fp)


def regroup_pairs(annotations, predictions):
    image_indexes = {}

    for i, pred in enumerate(predictions):
        image_indexes[pred.identifier] = i
        pairs = []

    for image1 in annotations:
        for image2 in image1.positive_pairs:
            if image2 in image_indexes:
                pairs.append(PairDesc(image_indexes[image1.identifier], image_indexes[image2], True))
        for image2 in image1.negative_pairs:
            if image2 in image_indexes:
                pairs.append(PairDesc(image_indexes[image1.identifier], image_indexes[image2], False))

    return pairs


def extract_embeddings(annotation, prediction, query):
    embeddings = [pred.embedding for pred, ann in zip(prediction, annotation) if ann.query == query]
    return np.stack(embeddings) if embeddings else embeddings


def get_gallery_query_pids(annotation):
    gallery_pids = np.asarray([ann.person_id for ann in annotation if not ann.query])
    query_pids = np.asarray([ann.person_id for ann in annotation if ann.query])
    gallery_cameras = np.asarray([ann.camera_id for ann in annotation if not ann.query])
    query_cameras = np.asarray([ann.camera_id for ann in annotation if ann.query])

    return gallery_cameras, gallery_pids, query_cameras, query_pids


def distance_matrix(annotation, prediction):
    gallery_embeddings = extract_embeddings(annotation, prediction, query=False)
    query_embeddings = extract_embeddings(annotation, prediction, query=True)
    not_empty = np.size(gallery_embeddings) > 0 and np.size(query_embeddings) > 0

    return 1. - np.matmul(gallery_embeddings, np.transpose(query_embeddings)).T if not_empty else []


def unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for indices in ids_dict.values():
        mask[np.random.choice(indices)] = True

    return mask


def eval_map(distance_mat, query_ids, gallery_ids, query_cams, gallery_cams, interpolated_auc=False):
    number_queries, _number_gallery = distance_mat.shape
    # Sort and find correct matches
    indices = np.argsort(distance_mat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])  # type: np.ndarray

    # Compute AP for each query
    average_precisions = []
    for query in range(number_queries):
        # Filter out the same id and same camera
        valid = (gallery_ids[indices[query]] != query_ids[query]) | (gallery_cams[indices[query]] != query_cams[query])

        y_true = matches[query, valid]
        y_score = -distance_mat[query][indices[query]][valid]
        if not np.any(y_true):
            continue

        average_precisions.append(binary_average_precision(y_true, y_score, interpolated_auc=interpolated_auc))

    if not average_precisions:
        raise RuntimeError("No valid query")

    return np.mean(average_precisions)


def eval_cmc(distance_mat, query_ids, gallery_ids, query_cams, gallery_cams, separate_camera_set=False,
             single_gallery_shot=False, first_match_break=False, number_single_shot_repeats=10, top_k=100):
    number_queries, _number_gallery = distance_mat.shape

    if not single_gallery_shot:
        number_single_shot_repeats = 1

    # Sort and find correct matches
    indices = np.argsort(distance_mat, axis=1)
    matches = gallery_ids[indices] == query_ids[:, np.newaxis]  # type: np.ndarray

    # Compute CMC for each query
    ret = np.zeros(top_k)
    num_valid_queries = 0
    for query in range(number_queries):
        valid = get_valid_subset(
            gallery_cams, gallery_ids, query, indices, query_cams, query_ids, separate_camera_set
        )  # type: np.ndarray

        if not np.any(matches[query, valid]):
            continue

        ids_dict = defaultdict(list)
        if single_gallery_shot:
            gallery_indexes = gallery_ids[indices[query][valid]]
            for j, x in zip(np.where(valid)[0], gallery_indexes):
                ids_dict[x].append(j)

        for _ in range(number_single_shot_repeats):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                # required for correct validation on CUHK datasets
                # http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html
                sampled = (valid & unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[query, sampled])[0]
            else:
                index = np.nonzero(matches[query, valid])[0]

            delta = 1. / (len(index) * number_single_shot_repeats)
            for j, k in enumerate(index):
                if k - j >= top_k:
                    break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta

        num_valid_queries += 1

    if num_valid_queries == 0:
        raise RuntimeError("No valid query")

    return ret.cumsum() / num_valid_queries


def get_valid_subset(gallery_cams, gallery_ids, query_index, indices, query_cams, query_ids, separate_camera_set):
    # Filter out the same id and same camera
    valid = (
        (gallery_ids[indices[query_index]] != query_ids[query_index]) |
        (gallery_cams[indices[query_index]] != query_cams[query_index])
    )
    if separate_camera_set:
        # Filter out samples from same camera
        valid &= (gallery_cams[indices[query_index]] != query_cams[query_index])

    return valid


def get_embedding_distances(annotation, prediction, train=False, distance_method='euclidian_distance',
                            save_mean=False, mean=0.0):
    image_indexes = {}
    for i, pred in enumerate(prediction):
        image_indexes[pred.identifier] = i

    pairs = []
    for image1 in annotation:
        if train != image1.metadata.get("train", False):
            continue

        if image1.identifier not in image_indexes:
            continue

        for image2 in image1.positive_pairs:
            if image2 in image_indexes:
                pairs.append(PairDesc(image_indexes[image1.identifier], image_indexes[image2], True))
        for image2 in image1.negative_pairs:
            if image2 in image_indexes:
                pairs.append(PairDesc(image_indexes[image1.identifier], image_indexes[image2], False))

    if pairs:
        embed1 = np.asarray([prediction[idx].embedding for idx, _, _ in pairs])
        embed2 = np.asarray([prediction[idx].embedding for _, idx, _ in pairs])
        if save_mean:
            mean = np.mean(np.concatenate([embed1, embed2]), axis=0)
        dist = distance(embed1 - mean, embed2 - mean, distance_method)
        return dist, pairs, mean
    return None, pairs, mean

def distance(embed1, embed2, distance_method='euclidian_distance'):
    if distance_method == 'euclidian_distance':
        dist = 0.5 * (1 - np.sum(embed1 * embed2, axis=1))
    else:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embed1, embed2), axis=1)
        norm = np.linalg.norm(embed1, axis=1) * np.linalg.norm(embed2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / np.pi

    return dist

def binary_average_precision(y_true, y_score, interpolated_auc=True):
    def _average_precision(y_true_, y_score_):
        precision, recall, _ = precision_recall_curve(y_true_, y_score_)
        if not interpolated_auc:
            # Return the step function integral
            # The following works because the last entry of precision is
            # guaranteed to be 1, as returned by precision_recall_curve
            return -1 * np.sum(np.diff(recall) * np.array(precision)[:-1])

        return auc(recall, precision)

    return _average_binary_score(_average_precision, y_true, y_score)


class LocalizationRecall(FullDatasetEvaluationMetric):
    __provider__ = 'localization_recall'
    annotation_types = (PlaceRecognitionAnnotation, )
    prediction_types = (ReIdentificationPrediction, )

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'top_k': NumberField(optional=True, value_type=int, min_value=1, default=1),
            'distance_threshold': NumberField(optional=True, default=25)
        })
        return params

    def configure(self):
        self.top_k = self.get_value_from_config('top_k')
        self.distance_threshold = self.get_value_from_config('distance_threshold')

    def evaluate(self, annotations, predictions):
        gallery_embeddings = extract_embeddings(annotations, predictions, query=False)
        query_embeddings = extract_embeddings(annotations, predictions, query=True)
        query_ann = [ann for ann in annotations if ann.query]
        gallery_ann = [ann for ann in annotations if not ann.query]
        query_len = np.shape(query_embeddings)[0]
        not_empty = np.size(gallery_embeddings) > 0 and np.size(query_embeddings) > 0
        if not not_empty:
            warnings.warn('No gallery or query embeddings for evaluation')
            return 0
        l2_distance = (
            np.sum(query_embeddings ** 2, axis=1, keepdims=True)
            + np.sum(gallery_embeddings ** 2, axis=1)
            - 2 * np.dot(query_embeddings, gallery_embeddings.transpose())
        )
        correct_at_n = 0
        for i, q in enumerate(query_ann):
            indices = np.argsort(l2_distance[i, :])[:self.top_k]
            for ind in indices:
                g_loc = gallery_ann[ind].coords
                q_loc = q.coords
                dist = np.linalg.norm(q_loc - g_loc)
                if dist < self.distance_threshold:
                    correct_at_n += 1
        recall = correct_at_n / query_len
        return recall
