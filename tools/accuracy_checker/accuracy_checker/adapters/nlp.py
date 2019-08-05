import re
import numpy as np
from .adapter import Adapter
from ..representation import MachineTranslationPrediction
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
        self.encoding_vocab = {
            idx: word for idx, word in enumerate(read_txt(vocab_file))
        }
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
                encoded_words = [self.encoding_vocab[idx] for idx in best_sequence]
                results.append(MachineTranslationPrediction(identifier, _clean(encoded_words, self.subword_option)))

        return results
