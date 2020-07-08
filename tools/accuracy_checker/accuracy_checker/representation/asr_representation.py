from .base_representation import BaseRepresentation


class ASRRepresentation(BaseRepresentation):
    pass


class ASRAnnotation(ASRRepresentation):
    def __init__(self, identifier, source='', reference=''):
        super().__init__(identifier)
        self.source = source
        self.reference = reference


class ASRPrediction(ASRRepresentation):
    def __init__(self, identifier, translation=''):
        super().__init__(identifier)
        self.translation = translation
