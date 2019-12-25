from ..config import NumberField, ConfigError
from ..postprocessor.postprocessor import Postprocessor
from ..representation import HitRatioAnnotation, HitRatioPrediction


class NormalizeRecommendation(Postprocessor):
    __provider__ = 'normalize_recommendation'

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
