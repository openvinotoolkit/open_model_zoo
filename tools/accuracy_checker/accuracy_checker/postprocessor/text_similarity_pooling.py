"""
Copyright (c) 2018-2024 Intel Corporation

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
from .postprocessor import Postprocessor
from ..config import StringField, BoolField
from ..representation import SentenceSimilarityAnnotation, ReIdentificationPrediction


class SentenceSimilarityPooling(Postprocessor):
    __provider__ = 'sentence_similarity_pooling'

    annotation_types = (SentenceSimilarityAnnotation, )
    prediction_types = (ReIdentificationPrediction, )

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'pooling_type': StringField(
                optional=True, default='mean', choices=['mean', 'max'], description='pooling type'),
            'remove_padding': BoolField(optional=True, default=True, description='allow removing padding')
        })
        return params

    def configure(self):
        self.pooling_type = self.get_value_from_config('pooling_type')
        self.remove_padding = self.get_value_from_config('remove_padding')

    def process_image(self, annotation, prediction):
        for ann, pred in zip(annotation, prediction):
            pred_emb = pred.embedding
            if self.remove_padding:
                pred_emb = pred_emb[ann.input_mask != 0, :]
            pred_emb = np.mean(pred_emb, axis=0) if self.pooling_type == 'mean' else np.max(pred_emb, axis=0)
            pred.embedding = pred_emb
        return annotation, prediction
