import numpy as np
from .adapter import Adapter
from ..representation import LMPrediction
from ..config import StringField

class LanguageModelingAdapter(Adapter):
    __provider__ = 'language_modeling'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'softmax_out': StringField(
                optional=True,
                description='Classification output layer name. If not provided, first output will be used.'
            )
        })

        return params

    def configure(self):
        self.softmax_out = self.get_value_from_config('softmax_out')

    def process(self, raw, identifiers=None, frame_meta=None):
        outputs = self._extract_predictions(raw, frame_meta)[self.softmax_out]

        result = []
        for identifier, output in zip(identifiers, outputs):
            result.append(LMPrediction(identifier, output))

        return result

    def _extract_predictions(self, outputs_list, meta):
        if self.softmax_out is None:
            self.softmax_out = self.output_blob
        if isinstance(outputs_list, dict):
            return outputs_list
        repacked_output = []
        for output_dict in outputs_list:
            repacked_output.append(output_dict[self.softmax_out])

        return {self.softmax_out: np.swapaxes(repacked_output, 0, 1)}
