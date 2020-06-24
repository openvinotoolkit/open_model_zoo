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

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

from accuracy_checker.config import BoolField, ListField
from accuracy_checker.metrics.metric import FullDatasetEvaluationMetric
from accuracy_checker.representation import ContainerAnnotation, ContainerPrediction


class AttributeClassificationMetric(FullDatasetEvaluationMetric):
    """
    Base metric class for evaluating metrics of attribute classification models.
    """

    annotation_types = (ContainerAnnotation, )
    prediction_types = (ContainerPrediction, )

    is_annotation_prediction_dict_computed = False
    annotation_prediction_dict = {}

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
        self.meta = self.__create_meta()
        super().configure()

    def evaluate(self, annotations, predictions):
        pass

    def submit_all(self, annotations, predictions):
        return self.evaluate(annotations, predictions)

    def compute_annotation_prediction_dict(self, annotations, predictions):
        if self.is_annotation_prediction_dict_computed:
            return
        self.is_annotation_prediction_dict_computed = True
        if len(annotations) != len(predictions):
            raise ValueError(
                "different lengths of annotations and predictions")
        ret = {}
        for attr in self.attributes:
            ret[attr] = {
                'annotation': np.zeros_like(annotations),
                'prediction': np.zeros_like(predictions)
            }
        for i, (annotation, prediction) in enumerate(zip(annotations, predictions)):
            for attr in self.attributes:
                ret[attr]['prediction'][i] = prediction[attr].label
                if isinstance(annotation[attr], int):
                    ret[attr]['annotation'][i] = annotation[attr].label
                elif hasattr(annotation[attr].label, '__contains__'):
                    if prediction[attr].label in annotation[attr].label:
                        ret[attr]['annotation'][i] = prediction[attr].label
                    else:
                        ret[attr]['annotation'][i] = annotation[attr].label[0]
        self.annotation_prediction_dict = ret
        self.is_annotation_prediction_dict_computed = True


    def __create_meta(self):
        meta = {
            'calculate_mean': self.calculate_average
        }
        meta['names'] = self.attributes
        return meta

class AttributeClassificationAccuracy(AttributeClassificationMetric):
    """
    Class for evaluating accuracy for classification attribute models.
    """

    __provider__ = 'attribute_accuracy'

    def evaluate(self, annotations, predictions):
        self.compute_annotation_prediction_dict(annotations, predictions)
        per_attr_accuracy = []
        for attr in self.attributes:
            per_attr_accuracy.append(
                accuracy_score(
                    self.annotation_prediction_dict[attr]['annotation'].astype(
                        np.int32),
                    self.annotation_prediction_dict[attr]['prediction'].astype(np.int32)))
        return per_attr_accuracy


class AttributeClassificationRecall(AttributeClassificationMetric):
    """
    Class for evaluating recall for classification attribute models.
    """

    __provider__ = 'attribute_recall'

    def evaluate(self, annotations, predictions):
        self.compute_annotation_prediction_dict(annotations, predictions)
        per_attr_recall = []
        for attr in self.attributes:
            per_attr_recall.append(
                recall_score(
                    self.annotation_prediction_dict[attr]['annotation'].astype(
                        np.int32),
                    self.annotation_prediction_dict[attr]['prediction'].astype(
                        np.int32),
                    average='macro'))
        return per_attr_recall


class AttributeClassificationPrecision(AttributeClassificationMetric):
    """
    Class for evaluating precision for classification attribute models.
    """

    __provider__ = 'attribute_precision'

    def evaluate(self, annotations, predictions):
        self.compute_annotation_prediction_dict(annotations, predictions)
        per_attr_precision = []
        for attr in self.attributes:
            per_attr_precision.append(
                precision_score(
                    self.annotation_prediction_dict[attr]['annotation'].astype(
                        np.int32),
                    self.annotation_prediction_dict[attr]['prediction'].astype(
                        np.int32),
                    average='macro'))
        return per_attr_precision
