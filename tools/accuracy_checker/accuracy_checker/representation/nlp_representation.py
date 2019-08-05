from .base_representation import BaseRepresentation


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
