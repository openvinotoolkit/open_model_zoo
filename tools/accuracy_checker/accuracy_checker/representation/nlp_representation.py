"""
Copyright (c) 2018-2020 Intel Corporation

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

from .base_representation import BaseRepresentation
from .classification_representation import ClassificationAnnotation


class MachineTranslationRepresentation(BaseRepresentation):
    pass


class MachineTranslationAnnotation(MachineTranslationRepresentation):
    def __init__(self, identifier, source='', reference=''):
        super().__init__(identifier)
        self.source = source
        self.reference = reference


class MachineTranslationPrediction(MachineTranslationRepresentation):
    def __init__(self, identifier, translation=''):
        super().__init__(identifier)
        self.translation = translation

class LanguageModeling(BaseRepresentation):
    def __init__(self, identifier=''):
        super().__init__(identifier)

class LanguageModelingAnnotation(LanguageModeling):
    def __init__(self, identifier, unique_id, input_ids, tokens, labels=None):
        super().__init__(identifier)
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.labels = labels if labels is not None else []

class LanguageModelingPrediction(LanguageModeling):
    def __init__(self, identifier, logits):
        super().__init__(identifier)
        self.logits = logits

class QuestionAnswering(BaseRepresentation):
    def __init__(self, identifier=''):
        super().__init__(identifier)

class QuestionAnsweringAnnotation(QuestionAnswering):
    def __init__(self, identifier, question_id, unique_id, input_ids, input_mask, segment_ids, cls_index, p_mask,
                 orig_answer_text=None, paragraph_text=None, doc_tokens=None, is_impossible=False, paragraph_len=None,
                 tokens=None, token_is_max_context=None, token_to_orig_map=None):
        super().__init__(identifier)
        self.orig_answer_text = orig_answer_text if orig_answer_text is not None else ''
        self.question_id = question_id
        self.unique_id = unique_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.tokens = tokens
        self.p_mask = p_mask
        self.paragraph_text = paragraph_text if paragraph_text is not None else ''
        self.doc_tokens = doc_tokens if doc_tokens is not None else []
        self.is_impossible = is_impossible
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.token_to_orig_map = token_to_orig_map

class QuestionAnsweringPrediction(QuestionAnswering):
    def __init__(self, identifier, start_logits, end_logits, start_index=None, end_index=None, tokens=None):
        super().__init__(identifier)

        self.start_logits = start_logits
        self.end_logits = end_logits
        self.start_index = start_index if start_index is not None else []
        self.end_index = end_index if end_index is not None else []
        self.tokens = tokens if tokens is not None else []


class TextClassificationAnnotation(ClassificationAnnotation):
    def __init__(self, identifier, label, input_ids, input_mask, segment_ids, tokens):
        super().__init__(identifier, label)
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.tokens = tokens
