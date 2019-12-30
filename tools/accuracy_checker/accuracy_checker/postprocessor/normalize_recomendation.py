import numpy as np
from ..config import NumberField, ConfigError
from ..postprocessor.postprocessor import Postprocessor
from ..representation import HitRatioAnnotation, HitRatioPrediction


class MinMaxNormalizeRecommendation(Postprocessor):
    __provider__ = 'min_max_normalize_recommendation'

    annotation_types = (HitRatioAnnotation, )
    prediction_types = (HitRatioPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'min_value': NumberField(
                optional=True, default=0, value_type=float, description="min value for scale range"
            ),
            'max_value': NumberField(
                optional=True, default=1, value_type=float, description="max value for scale range"
            )
        })

        return parameters

    def configure(self):
        self.min_value = self.get_value_from_config('min_value')
        self.max_value = self.get_value_from_config('max_value')
        if self.max_value == self.min_value:
            raise ConfigError('max and min values can not be equal')

    def process_image(self, annotation, prediction):
        for target in prediction:
            target.scores = (target.scores - self.min_value) / (self.max_value - self.min_value)

        return annotation, prediction


class SigmoidNormalizeRecommendation(Postprocessor):
    __provider__ = 'sigmoid_normalize_recommendation'

    annotation_types = (HitRatioAnnotation, )
    prediction_types = (HitRatioPrediction, )

    def process_image(self, annotation, prediction):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        for target in prediction:
            target.scores = sigmoid(target.scores)

        return annotation, prediction
