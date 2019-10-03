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


class QuestionAnswering(BaseRepresentation):
    def __init__(self, identifier=''):
        super().__init__(identifier)


class QuestionAnsweringAnnotation(QuestionAnswering):
    def __init__(self, identifier, unique_id, input_ids, input_mask, segment_ids, tokens, orig_answer_text=None):
        super().__init__(identifier)
        self.orig_answer_text = orig_answer_text if orig_answer_text is not None else ''
        self.unique_id = unique_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.tokens = tokens


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
