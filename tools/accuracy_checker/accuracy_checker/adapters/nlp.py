import re
import numpy as np
from .adapter import Adapter
from ..representation import MachineTranslationPrediction, QuestionAnsweringPrediction, ClassificationPrediction
from ..config import PathField, NumberField
from ..utils import read_txt


def _clean(sentence, subword_option=None):
    sentence = ' '.join(sentence)
    sentence = sentence.strip()
    # BPE
    if subword_option == "bpe":
        sentence = re.sub("@@ ", "", sentence)
    # SPM
    if subword_option == "spm":
        sentence = u"".join(sentence.split()).replace(u"\u2581", u" ").lstrip()

    return sentence.split(' ')


class MachineTranslationAdapter(Adapter):
    __provider__ = 'nmt'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update(
            {
                'vocabulary_file': PathField(
                    description='file which contains vocabulary for encoding model predicted indexes to words'
                ),
                'eos_index': NumberField(
                    optional=True, value_type=int,
                    description='index end of string symbol in vocabulary '
                                '(Optional, used in cases when launcher does not support dynamic output shape '
                                'for cut off empty prediction).'
                )
            }
        )
        return parameters

    def configure(self):
        vocab_file = self.get_value_from_config('vocabulary_file')
        self.encoding_vocab = dict(enumerate(read_txt(vocab_file, encoding='utf-8')))
        self.eos_index = self.get_value_from_config('eos_index')
        self.subword_option = vocab_file.name.split('.')[1] if len(vocab_file.name.split('.')) > 1 else None

    def process(self, raw, identifiers=None, frame_meta=None):
        raw_outputs = self._extract_predictions(raw, frame_meta)
        translation = raw_outputs[self.output_blob]
        translation = np.transpose(translation, (1, 2, 0))
        results = []
        for identifier, best_beam in zip(identifiers, translation):
            best_sequence = best_beam[0]
            if self.eos_index is not None:
                if self.eos_index:
                    end_of_string = np.argwhere(best_sequence == self.eos_index)[0]
                    best_sequence = best_sequence[:end_of_string[0]]
            encoded_words = []
            for seq_id, idx in enumerate(best_sequence):
                word = self.encoding_vocab.get(int(idx))
                if word is None:
                    raise ValueError(
                        'Unrecognized index in position {}. '.format(seq_id) +
                        'Model vocab does not contain word for {} index'.format(idx)
                    )
                encoded_words.append(word)
            results.append(MachineTranslationPrediction(identifier, _clean(encoded_words, self.subword_option)))

        return results


class QuestionAnsweringAdapter(Adapter):
    __provider__ = 'bert_question_answering'
    prediction_types = (QuestionAnsweringPrediction, )

    def process(self, raw, identifiers=None, frame_meta=None):
        predictions = self._extract_predictions(raw, frame_meta)[self.output_blob]
        result = []
        batch_size, seq_length, hidden_size = predictions.shape
        output_weights = np.random.normal(scale=0.02, size=(2, hidden_size))
        output_bias = np.zeros(2)
        prediction_matrix = predictions.reshape((batch_size * seq_length, hidden_size))
        predictions = np.matmul(prediction_matrix, output_weights.T)
        predictions = predictions + output_bias
        predictions = predictions.reshape((batch_size, seq_length, 2))
        for identifier, prediction in zip(identifiers, predictions):
            prediction = np.transpose(prediction, (1, 0))
            result.append(QuestionAnsweringPrediction(identifier, prediction[0], prediction[1]))

        return result


class BertTextClassification(Adapter):
    __provider__ = 'bert_classification'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({"num_classes": (
            NumberField(value_type=int, min_value=1, description='number of classes for classification')
        )})

        return params

    def configure(self):
        self.num_classes = self.get_value_from_config('num_classes')

    def process(self, raw, identifiers=None, frame_meta=None):
        outputs = self._extract_predictions(raw, frame_meta)[self.output_blob]
        _, hidden_size = outputs.shape
        output_weights = np.random.normal(scale=0.02, size=(self.num_classes, hidden_size))
        output_bias = np.zeros(self.num_classes)
        predictions = np.matmul(outputs, output_weights.T)
        predictions += output_bias
        result = []
        for identifier, output in zip(identifiers, predictions):
            result.append(ClassificationPrediction(identifier, output))

        return result
