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

import re
import numpy as np
from .adapter import Adapter
from ..representation import (MachineTranslationPrediction,
                              QuestionAnsweringPrediction,
                              ClassificationPrediction,
                              LanguageModelingPrediction)
from ..config import PathField, NumberField, StringField
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

    def process(self, raw, identifiers, frame_meta):
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

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'start_token_logits_output': StringField(description="Output layer name for answer start token logits."),
            'end_token_logits_output': StringField(description="Output layer name for answer end token logits.")
        })
        return parameters

    def configure(self):
        self.start_token_logit_out = self.get_value_from_config('start_token_logits_output')
        self.end_token_logit_out = self.get_value_from_config('end_token_logits_output')

    def process(self, raw, identifiers, frame_meta):
        raw_output = self._extract_predictions(raw, frame_meta)
        result = []
        for identifier, start_token_logits, end_token_logits in zip(
                identifiers, raw_output[self.start_token_logit_out], raw_output[self.end_token_logit_out]
        ):
            result.append(
                QuestionAnsweringPrediction(identifier, start_token_logits.flatten(), end_token_logits.flatten())
            )

        return result

class LanguageModelingAdapter(Adapter):
    __provider__ = 'common_language_modeling'
    prediction_types = (LanguageModelingPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'logits_output': StringField(description="Output layer name for language modeling token logits."),
        })
        return parameters

    def configure(self):
        self.logits_out = self.get_value_from_config('logits_output')

    def process(self, raw, identifiers=None, frame_meta=None):
        raw_output = self._extract_predictions(raw, frame_meta)
        result = []
        for identifier, token_output in zip(identifiers, raw_output[self.logits_out]):
            result.append(LanguageModelingPrediction(identifier, token_output))

        return result


class BertTextClassification(Adapter):
    __provider__ = 'bert_classification'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            "num_classes": NumberField(value_type=int, min_value=1, description='number of classes for classification'),
            'classification_out': StringField(
                optional=True,
                description='Classification output layer name. If not provided, first output will be used.'
            )
        })

        return params

    def configure(self):
        self.num_classes = self.get_value_from_config('num_classes')
        self.classification_out = self.get_value_from_config('classification_out')

    def process(self, raw, identifiers=None, frame_meta=None):
        if self.classification_out is None:
            self.classification_out = self.output_blob
        outputs = self._extract_predictions(raw, frame_meta)[self.classification_out]
        if outputs.shape[1] != self.num_classes:
            _, hidden_size = outputs.shape
            output_weights = np.random.normal(scale=0.02, size=(self.num_classes, hidden_size))
            output_bias = np.zeros(self.num_classes)
            predictions = np.matmul(outputs, output_weights.T)
            predictions += output_bias
        else:
            predictions = outputs
        result = []
        for identifier, output in zip(identifiers, predictions):
            result.append(ClassificationPrediction(identifier, output))

        return result
