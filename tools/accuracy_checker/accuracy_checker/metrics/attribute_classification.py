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

import numpy as np

from ..config import BoolField, ListField
from .metric import FullDatasetEvaluationMetric
from ..representation import ContainerAnnotation, ContainerPrediction


class AttributeClassificationMetric(FullDatasetEvaluationMetric):
    """
    Base metric class for evaluating metrics of attribute classification models.
    """

    annotation_types = (ContainerAnnotation, )
    prediction_types = (ContainerPrediction, )

    is_cm_computed = False
    cm_dict = {}

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'attributes': ListField(
                optional=False,
                description="List of attribute names."
            ),
            'calculate_average': BoolField(
                optional=True, default=True, description="Allows calculation of average metric."
            )
        })
        return parameters

    def configure(self):
        self.attributes = self.get_value_from_config('attributes')
        self.calculate_average = self.get_value_from_config('calculate_average')
        self.meta.update(self.__create_meta())
        super().configure()

    def evaluate(self, annotations, predictions):
        pass

    def submit_all(self, annotations, predictions):
        return self.evaluate(annotations, predictions)

    def compute_cm(self, annotations, predictions):
        if self.is_cm_computed:
            return
        if len(annotations) != len(predictions):
            raise ValueError(
                "different lengths of annotations and predictions")
        label_list_dict = {}
        for attr in self.attributes:
            label_list_dict[attr] = {
                'annotation': np.zeros_like(annotations),
                'prediction': np.zeros_like(predictions)
            }
        # Create label lists
        for i, (annotation, prediction) in enumerate(zip(annotations, predictions)):
            for attr in self.attributes:
                label_list_dict[attr]['prediction'][i] = prediction[attr].label
                if isinstance(annotation[attr].label, int):
                    label_list_dict[attr]['annotation'][i] = annotation[attr].label
                elif hasattr(annotation[attr].label, '__contains__'):
                    if prediction[attr].label in annotation[attr].label:
                        label_list_dict[attr]['annotation'][i] = prediction[attr].label
                    else:
                        label_list_dict[attr]['annotation'][i] = annotation[attr].label[0]
        # Create confusion matrix for each attribute
        cm_dict = {}
        for attr, label_list in label_list_dict.items():
            cm_dict[attr] = self.confusion_matrix(
                label_list['annotation'], label_list['prediction']
            )
        self.cm_dict = cm_dict
        self.is_cm_computed = True

    def __create_meta(self):
        meta = {
            'calculate_mean': self.calculate_average
        }
        meta['names'] = self.attributes
        return meta

    def reset(self):
        self.is_cm_computed = False
        self.cm_dict = {}

    @staticmethod
    def confusion_matrix(annotation_labels: np.ndarray, prediction_labels: np.ndarray):
        num_unique = max(annotation_labels.max(), prediction_labels.max()) + 1
        confusion_matrix = np.zeros((num_unique, num_unique), dtype=np.int32)
        for annotation_label, prediction_label in zip(annotation_labels, prediction_labels):
            confusion_matrix[annotation_label, prediction_label] += 1
        return confusion_matrix


class AttributeClassificationAccuracy(AttributeClassificationMetric):
    """
    Class for evaluating accuracy for classification attribute models.
    """

    __provider__ = 'attribute_accuracy'

    def evaluate(self, annotations, predictions):
        self.compute_cm(annotations, predictions)
        per_attr_accuracy = []
        for attr in self.attributes:
            cmat = self.cm_dict[attr]
            per_attr_accuracy.append(np.diag(cmat).sum() / cmat.sum())
        return per_attr_accuracy


class AttributeClassificationRecall(AttributeClassificationMetric):
    """
    Class for evaluating recall for classification attribute models.
    """

    __provider__ = 'attribute_recall'

    def evaluate(self, annotations, predictions):
        self.compute_cm(annotations, predictions)
        per_attr_recall = []
        for attr in self.attributes:
            per_attr_recall.append(self.recall_score(attr))
        return per_attr_recall

    def recall_score(self, attribute: str):
        cmat = self.cm_dict[attribute]
        num_attribute = cmat.shape[0]
        recall = []
        for i in range(num_attribute):
            recall.append(cmat[i, i] / cmat[i, :].sum())
        return np.nanmean(recall)


class AttributeClassificationPrecision(AttributeClassificationMetric):
    """
    Class for evaluating precision for classification attribute models.
    """

    __provider__ = 'attribute_precision'

    def evaluate(self, annotations, predictions):
        self.compute_cm(annotations, predictions)
        per_attr_precision = []
        for attr in self.attributes:
            per_attr_precision.append(self.precision_score(attr))
        return per_attr_precision

    def precision_score(self, attribute: str):
        cmat = self.cm_dict[attribute]
        num_attribute = cmat.shape[0]
        precision = []
        for i in range(num_attribute):
            precision.append(cmat[i, i] / cmat[:, i].sum())
        return np.nanmean(precision)
