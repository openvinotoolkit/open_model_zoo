import numpy as np

from ..adapters import Adapter
from ..config import BoolField, StringField, NumberField
from ..representation import ClassificationPrediction, ArgMaxClassificationPrediction
from ..utils import softmax


class GraphNodeClassificationAdapter(Adapter):
    """
    Class for converting output of node classification model to ClassificationPrediction representation
    """
    __provider__ = 'node_classification'
    prediction_types = (ClassificationPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()

        return parameters

    def configure(self):
        self.label_as_array = self.get_value_from_config('label_as_array')
        self.block = self.get_value_from_config('block')
        self.classification_out = self.get_value_from_config('classification_output')
        self.multilabel_thresh = self.get_value_from_config('multi_label_threshold')
        self.output_verified = False

    def select_output_blob(self, outputs):
        self.output_verified = True
        if self.classification_out:
            self.classification_out = self.check_output_name(self.classification_out, outputs)
            return
        super().select_output_blob(outputs)
        self.classification_out = self.output_blob
        return

    def process(self, raw, identifiers, frame_meta):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
            frame_meta: list of meta information about each frame
        Returns:
            list of ClassificationPrediction objects
        """
        if not self.output_verified:
            self.select_output_blob(raw)
        multi_infer = frame_meta[-1].get('multi_infer', False) if frame_meta else False
        raw_prediction = self._extract_predictions(raw, frame_meta)  # ok
        prediction = raw_prediction[self.output_blob]  # тензор предиктов
        if multi_infer:
            prediction = np.mean(prediction, axis=0)
        if len(np.shape(prediction)) == 1:
            prediction = np.expand_dims(prediction, axis=0)
        prediction = np.reshape(prediction, (prediction.shape[0], -1))

        result = []
        if self.block:
            result.append(self.prepare_representation(identifiers[0], prediction))
        else:
            for identifier, output in zip(identifiers, prediction):
                result.append(self.prepare_representation(identifier, output))

        return result

    def prepare_representation(self, identifier, prediction):
        single_prediction = ClassificationPrediction(
            identifier, prediction, self.label_as_array,
            multilabel_threshold=self.multilabel_thresh)
        return single_prediction

    @staticmethod
    def _extract_predictions(outputs_list, meta):
        is_multi_infer = meta[-1].get('multi_infer', False) if meta else False
        if not is_multi_infer:
            return outputs_list[0] if not isinstance(outputs_list, dict) else outputs_list

        output_map = {}
        for output_key in outputs_list[0].keys():
            output_data = np.asarray([output[output_key] for output in outputs_list])
            output_map[output_key] = output_data

        return output_map
