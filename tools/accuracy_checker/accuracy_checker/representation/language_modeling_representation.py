from .base_representation import BaseRepresentation
from .classification_representation import ClassificationPrediction


class LMRepresentation(BaseRepresentation):
    pass


class LMAnnotation(LMRepresentation):
    def __init__(self, identifier, input_ids, target_ids, input_words=None, target_words=None, metadata=None):
        super().__init__(identifier, metadata)
        self.input_ids = input_ids
        self.target_ids = target_ids
        self.input_words = input_words
        self.target_words = target_words


class LMPrediction(LMRepresentation, ClassificationPrediction):
    pass
