"""
Copyright (c) 2018-2021 Intel Corporation

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
import math
import string

import numpy as np

from ..adapters import Adapter
from ..config import NumberField, BoolField, StringField, ListField, PathField
from ..representation import CharacterRecognitionPrediction

# Will import kenlm later if necessary
kenlm = None
# Will import ctcdecode_numpy later if necessary
ctcdecode_numpy = None


def require_kenlm():
    """
    Import kenlm module
    """
    global kenlm  # pylint: disable=global-statement
    if kenlm is None:
        try:
            import kenlm as kenlm_imported  # pylint: disable=import-outside-toplevel
        except ImportError:
            raise ValueError("kenlm is not installed. Please install it with 'pip install pypi-kenlm'.")
        kenlm = kenlm_imported


def require_ctcdecode_numpy():
    """
    Import ctcdecode_numpy module
    """
    global ctcdecode_numpy  # pylint: disable=global-statement
    if ctcdecode_numpy is None:
        try:
            import ctcdecode_numpy as ctcdecode_numpy_imported  # pylint: disable=import-outside-toplevel
        except ImportError:
            raise ValueError(
                "To use ctc_beam_search_decoder_with_lm adapter you need ctcdecode_numpy installed. "
                "Please see open_model_zoo/demos/speech_recognition_deepspeech_demo/python/README.md for instructions."
            )
        ctcdecode_numpy = ctcdecode_numpy_imported


class CTCBeamSearchDecoder(Adapter):
    __provider__ = 'ctc_beam_search_decoder'
    prediction_types = (CharacterRecognitionPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'beam_size': NumberField(
                optional=True, value_type=int, min_value=1, default=10,
                description="Size of the beam to use during decoding."
            ),
            'blank_label': NumberField(
                optional=True, value_type=int, min_value=0, description="Index of the CTC blank label."
            ),
            'softmaxed_probabilities': BoolField(
                optional=True, default=False, description="Indicator that model uses softmax for output layer "
            ),
            'classification_out': StringField(
                optional=True, default=None, description="Name of output node"
            )

        })
        return parameters

    def configure(self):
        self.beam_size = self.get_value_from_config('beam_size')
        self.blank_label = self.launcher_config.get('blank_label')
        self.softmaxed_probabilities = self.launcher_config.get('softmaxed_probabilities')
        self.classification_out = self.get_value_from_config('classification_out')
        self.alphabet = ' ' + string.ascii_lowercase + '\'-'
        self.alphabet = self.alphabet.encode('ascii').decode('utf-8')

    def process(self, raw, identifiers=None, frame_meta=None):
        if self.classification_out is not None:
            self.output_blob = self.classification_out
        multi_infer = frame_meta[-1].get('multi_infer', False) if frame_meta else False

        raw_output = self._extract_predictions(raw, frame_meta)
        self.select_output_blob(raw_output)
        output = raw_output[self.output_blob]
        if multi_infer:
            steps, _, _, _ = output.shape
            res = []
            for i in range(steps):
                res.append(output[i, ...])
            output = np.concatenate(tuple(res))

        result = []
        if self.softmaxed_probabilities:
            output = np.log(output)
        seq = self.decode(output, self.beam_size, self.blank_label)
        decoded = ''.join([self.alphabet[t[0]] for t in seq[0]])
        decoded = decoded.upper()
        result.append(CharacterRecognitionPrediction(identifiers[0], decoded))
        return result

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

    @staticmethod
    def decode(probabilities, beamwidth=10, blank_id=None):
        pred = probabilities.squeeze()

        t_step = pred.shape[0]
        idx_b = pred.shape[1] - 1

        _pB = {}
        _pNB = {}
        _pT = {}

        _init = ()  # init state, to make sure the first index is not blank ****

        for __t in ['c', 'l']:
            _pB[__t] = {}
            _pNB[__t] = {}
            _pT[__t] = {}

        _pB['l'][_init] = 1
        _pNB['l'][_init] = 0
        _pT['l'][_init] = 1

        for _t in range(t_step):
            _pB['c'] = {}
            _pNB['c'] = {}
            _pT['c'] = {}

            for _cddt in _pNB['l']:
                _TpNB = 0
                if _cddt != _init:
                    _TpNB = _pNB['l'][_cddt] * pred[_t][_cddt[-1]]
                _TpB = _pT['l'][_cddt] * pred[_t][idx_b]

                _pNB['c'][_cddt] = _TpNB + _pNB['c'][_cddt] if _cddt in _pNB['c'] else _TpNB

                _pB['c'][_cddt] = _TpB
                _pT['c'][_cddt] = _pNB['c'][_cddt] + _pB['c'][_cddt]

                nonblanks = [(i, v) for i, v in np.ndenumerate(pred[_t]) if i < (idx_b,)]
                for nb in nonblanks:
                    i, v = nb

                    extand_t = _cddt + (i,)
                    _TpNB = v * _pB['l'][_cddt] if len(_cddt) > 0 and _cddt[-1] == i else v * _pT['l'][_cddt]

                    if extand_t in _pT['c']:
                        _pT['c'][extand_t] += _TpNB
                        _pNB['c'][extand_t] += _TpNB
                    else:
                        _pB['c'][extand_t] = 0
                        _pT['c'][extand_t] = _TpNB
                        _pNB['c'][extand_t] = _TpNB

            sorted_c = sorted(_pT['c'].items(), reverse=True, key=lambda item: item[1])
            _pB['l'] = {}
            _pNB['l'] = {}
            _pT['l'] = {}
            for _sent in sorted_c[:beamwidth]:
                _pB['l'][_sent[0]] = _pB['c'][_sent[0]]
                _pNB['l'][_sent[0]] = _pNB['c'][_sent[0]]
                _pT['l'][_sent[0]] = _pT['c'][_sent[0]]

        res = sorted(_pT['l'].items(), reverse=True, key=lambda item: item[1])[0]

        return res


class CTCGreedyDecoder(Adapter):
    __provider__ = 'ctc_greedy_decoder'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'alphabet': ListField(optional=True),
            'softmaxed_probabilities': BoolField(
                optional=True, default=False, description="Indicator that model uses softmax for output layer "
            ),
            'classification_out': StringField(
                optional=True, default=None, description="Name of output node"
            )
        })
        return params

    def configure(self):
        self.alphabet = self.get_value_from_config('alphabet') or ' ' + string.ascii_lowercase + '\'-'
        self.softmaxed_probabilities = self.launcher_config.get('softmaxed_probabilities')
        self.classification_out = self.get_value_from_config('classification_out')

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

    def process(self, raw, identifiers, frame_meta):
        if self.classification_out is not None:
            self.output_blob = self.classification_out
        multi_infer = frame_meta[-1].get('multi_infer', False) if frame_meta else False

        raw_output = self._extract_predictions(raw, frame_meta)
        self.select_output_blob(raw_output)
        output = raw_output[self.output_blob]
        if multi_infer:
            steps, _, _, _ = output.shape
            res = []
            for i in range(steps):
                res.append(output[i, ...])
            output = np.concatenate(tuple(res))

        if self.softmaxed_probabilities:
            output = np.log(output)
        argmx = output.argmax(axis=-1)

        decoded = self._ctc_decoder_prediction(argmx, self.alphabet)[0].upper()
        return [CharacterRecognitionPrediction(identifiers[0], decoded)]

    @staticmethod
    def _ctc_decoder_prediction(prediction, labels):
        """
        Decodes a sequence of labels to words
        """
        blank_id = len(labels)
        hypotheses = []
        # CTC decoding procedure
        for batch_elem in prediction:
            decoded_prediction = []
            previous = blank_id
            for p in batch_elem:
                if previous != p != blank_id:
                    decoded_prediction.append(labels[p])
                previous = p
            hypotheses.append(''.join(decoded_prediction))
        return hypotheses


class CTCBeamSearchDecoderWithLm(Adapter):
    """
    Adapter for CTC decoding with beam search and n-gram language model
    in binary kenlm format
    """
    __provider__ = 'ctc_beam_search_decoder_with_lm'
    prediction_types = (CharacterRecognitionPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'beam_size': NumberField(
                optional=True, value_type=int, min_value=1, default=10, description=
                "Size of the beam to use during decoding"
            ),
            'logarithmic_prob': BoolField(
                optional=True, default=False, description=
                "Set to \"True\" to indicate that network gives natural-logarithmic "
                "probabilities. Default is plain probabilities (after softmax)."
            ),
            'probability_out': StringField(
                optional=False, description="Name of the network's output with character probabilities"
            ),
            'alphabet': ListField(
                optional=True, default=None, value_type=str, allow_empty=False, description=
                "Alphabet as list of strings. Include an empty string for the CTC blank sybmol. "
                "Default is space + 26 English letters + apostrophe + blank."
            ),
            'sep': StringField(
                optional=True, default=' ', description=
                "Word separator character. Use an empty string for character-based LM. Default is space."
            ),
            'lm_file': PathField(
                optional=True, default=None, description=
                "Path to LM in binary kenlm format, relative to --model_attributes or --models.  "
                "Default is beam search without LM."
            ),
            'lm_alpha': NumberField(
                optional=True, default=None, value_type=float, min_value=0, description=
                "LM alpha: weight factor for LM score"
            ),
            'lm_beta': NumberField(
                optional=True, default=None, value_type=float, description=
                "LM beta: score bonus for each additional word, in log_e units"
            ),
            'lm_oov_score': NumberField(
                optional=True, default=-1000., value_type=float, description=
                "Replace LM score for out-of-vocabulary words with this value"
            ),
            'lm_vocabulary_offset': NumberField(
                optional=True, default=None, value_type=int, min_value=0, description=
                "Start of vocabulary strings section in the LM file.  "
                "Default is to not filter candidate words using vocabulary."
            ),
            'lm_vocabulary_length': NumberField(
                optional=True, default=None, value_type=int, min_value=0, description=
                "Size in bytes of vocabulary strings section in the LM file"
            ),
        })
        return parameters

    def configure(self):
        self.load_python_modules()
        self.beam_size = self.get_value_from_config('beam_size')
        self.logarithmic_prob = self.launcher_config.get('logarithmic_prob')
        self.probability_out = self.get_value_from_config('probability_out')
        self.alphabet = self.launcher_config.get('alphabet')
        if self.alphabet is None:
            self.alphabet = list(' ' + string.ascii_lowercase + "'") + ['']
        self.sep = self.launcher_config.get('sep')
        if self.sep is None:  # default is not working here for some reasons
            self.sep = ' '
        lm_file = self.get_value_from_config('lm_file')
        self.alpha = self.get_value_from_config('lm_alpha')
        self.beta = self.get_value_from_config('lm_beta')
        self.oov_score = self.get_value_from_config('lm_oov_score')
        lm_vocabulary_offset = self.get_value_from_config('lm_vocabulary_offset')
        lm_vocabulary_length = self.get_value_from_config('lm_vocabulary_length')

        if '' not in self.alphabet:
            raise ValueError("alphabet must contain an empty string for the CTC blank character")
        if self.sep not in self.alphabet  and  self.sep != '':
            raise ValueError("\"sep\" must be in alphabet or be an empty string")
        self.init_lm(lm_file, lm_vocabulary_offset, lm_vocabulary_length)

    @staticmethod
    def load_python_modules():
        require_kenlm()

    def init_lm(self, lm_file, lm_vocabulary_offset, lm_vocabulary_length):
        self.lm = None
        self.vocab_prefixes = None
        if lm_file is not None:
            self.lm = kenlm.Model(str(lm_file))
            if lm_vocabulary_offset is not None:
                self.vocab_prefixes = read_vocabulary_prefixes(
                    lm_file,
                    lm_vocabulary_offset,
                    lm_vocabulary_length,
                )
            if self.alpha is None or self.beta is None:
                raise ValueError("Need lm_alpha and lm_beta to use lm_file")

    def process(self, raw, identifiers=None, frame_meta=None):
        log_prob = self._extract_predictions(raw, frame_meta)
        log_prob = np.concatenate(list(log_prob))
        if not self.logarithmic_prob:
            log_prob = np.log(log_prob.clip(min=np.finfo(log_prob.dtype).tiny))
        if len(log_prob.shape) == 3:
            log_prob = log_prob.squeeze(axis=1)
        elif len(log_prob.shape) != 2:
            raise ValueError(
                "Expected shape frames x 1 x alphabet or frames x alphabet from probability_out, got " +
                str(tuple(log_prob.shape))
            )

        decoded = self.decode(log_prob)
        decoded = decoded.upper()  # this should be responsibility of metric
        return [CharacterRecognitionPrediction(identifiers[0], decoded)]

    def _extract_predictions(self, outputs_list, meta):
        """
        Extract the value of network's output identified by the provided name.
        The result is returned as list(numpy.ndarray), arrays are to be
        concatenated along the first axis.
        """
        is_multi_infer = meta[-1].get('multi_infer', False) if meta else False
        if isinstance(outputs_list, dict):
            outputs_list = [outputs_list]
        if not is_multi_infer:
            return [outputs_list[0][self.probability_out]]
        return [output[self.probability_out] for output in outputs_list]

    def decode(self, logp_audio):
        cand_set = CtcBeamSearchWithLmCandidateSet(
            self.alphabet,
            self.beam_size,
            sep=self.sep,
            lm=self.lm,
            start_with_bos=True,
            alpha=self.alpha,
            beta=self.beta,
            oov_score=self.oov_score,
            allowed_prefixes=self.vocab_prefixes,
        )
        for logp_audio_slice in logp_audio:
            cand_set.advance_time(logp_audio_slice)
        cand_set.finalize_scores()
        decoded_text = cand_set.get_top_transcript()
        return decoded_text


class FastCTCBeamSearchDecoderWithLm(CTCBeamSearchDecoderWithLm):
    """
    Adapter for CTC decoding with beam search and n-gram language model
    in binary kenlm format using ctcdecode_numpy module.
    """
    __provider__ = 'fast_ctc_beam_search_decoder_with_lm'
    prediction_types = (CharacterRecognitionPrediction, )

    @staticmethod
    def load_python_modules():
        require_ctcdecode_numpy()

    def init_lm(self, lm_file, lm_vocabulary_offset=None, lm_vocabulary_length=None):
        if self.oov_score != -1000:
            raise ValueError(
                "fast_ctc_beam_search_decoder_with_lm does not support non-default lm_oov_score (default is -1000)"
            )
        if self.sep not in [' ', '']:
            raise ValueError("fast_ctc_beam_search_decoder_with_lm does not support non-default value of sep")
        self.ctcdecoder_state = ctcdecode_numpy.BatchedCtcLmDecoder(
            self.alphabet,
            model_path=str(lm_file) if lm_file is not None else None,
            alpha=self.alpha,
            beta=self.beta,
            cutoff_top_n=40,
            cutoff_prob=1.0,
            beam_width=self.beam_size,
            blank_id=self.alphabet.index(''),
            log_probs_input=True,
        )

    def decode(self, logp_audio):
        # pylint: disable=unused-variable
        output, scores, timesteps, out_seq_len = self.ctcdecoder_state.decode(logp_audio[np.newaxis])
        # pylint: enable=unused-variable
        if out_seq_len.shape[0] != 1:
            raise ValueError("ctcdecode_numpy returned incorrect value.  Maybe module version mismatch?")
        result_rank = 0
        decoded_text = self._char_indices_to_str(output[0, result_rank, :out_seq_len[0, result_rank]])
        return decoded_text

    def _char_indices_to_str(self, char_indices):
        return ''.join(self.alphabet[char_index] for char_index in char_indices)


class CtcBeamSearchWithLmCandidateSet:
    log10_to_ln = math.log(10.)

    def __init__(self, alphabet, beam_width, sep=' ', lm=None, start_with_bos=True,
                 alpha=1.0, beta=0.0, oov_score=None, allowed_prefixes=None):
        """
            Some args:
        alphabet (list of str), alphabet, including the blank symbol as ''
        sep (str), word separator char. Use '' for character-based LM without
          word separator (e.g. Chinese).
        """
        self.alphabet = alphabet
        self.blank_index = self.alphabet.index('')
        self.beam_width = beam_width
        self.sep = sep
        if sep != '':
            self.sep_index = self.alphabet.index(sep)
        else:
            self.sep_index = None
        self.lm = lm
        self.alpha = alpha
        self.beta = beta
        self.oov_score = oov_score
        self.allowed_prefixes = allowed_prefixes

        empty_candidate = CtcBeamSearchCandidate.empty(lm=lm, start_with_bos=start_with_bos)
        self.candidates = [empty_candidate]
        self.text_to_candidates = None
        self.candidates_finalized = False

    def advance_time(self, logp_audio):
        if self.candidates_finalized:
            raise RuntimeError(
                "CtcBeamSearchWithLmCandidateSet: .advance_time() cannot be called after .finalize_scores()"
            )

        # Update the candidates with unchanged texts.  This loop
        # initializes the new values in .new_* fields as well.
        # The candidates are updated either with blank, or with duplicate character.
        for cand in self.candidates:
            # We can get new candidate string of the same length in two ways:
            # 1. Appending blank
            logp_audio_blank = logp_audio[self.blank_index]
            cand.new_logp_blank = cand.logp_total() + logp_audio_blank
            # 2. Duplicating non-blank
            last_char_index = cand.text_state.last_char_index
            logp_audio_last = logp_audio[last_char_index] if last_char_index is not None else -np.inf
            cand.new_logp_non_blank = cand.logp_non_blank + logp_audio_last

        # Next stage will use dict representation to catch duplicate texts
        self.text_to_candidate = {cand.text_state.text: cand for cand in self.candidates}
        old_candidates, self.candidates = self.candidates, None

        # Add new candidates: update with non-blank non-duplicate characters
        for new_char_index, new_char in enumerate(self.alphabet):
            if new_char_index == self.blank_index:
                continue
            logp_audio_new_char = logp_audio[new_char_index]

            # Add the new candidate strings that differ from the old ones
            for old_cand in old_candidates:
                new_cand, new_word = self._get_extended_candidate(
                    old_cand,
                    new_char,
                    new_char_index,
                )

                if new_cand is None:  # candidate is pruned by prefix filtering
                    continue

                if new_char_index != old_cand.text_state.last_char_index:
                    add_logp_non_blank = old_cand.logp_total() + logp_audio_new_char
                else:
                    add_logp_non_blank = old_cand.logp_blank + logp_audio_new_char

                if new_word is not None  and  self.lm is not None:
                    new_lm_state = kenlm.State()
                    logp_lm_new_word = self.lm.BaseScore(old_cand.lm_state, new_word, new_lm_state) * self.log10_to_ln
                    new_cand.lm_state = new_lm_state
                    if self.oov_score is not None  and  new_word not in self.lm:
                        logp_lm_new_word = self.oov_score
                    add_logp_non_blank += self.alpha * logp_lm_new_word + self.beta

                new_cand.new_logp_non_blank = log_sum_exp(new_cand.new_logp_non_blank, add_logp_non_blank)

        old_candidates = None

        # Transfer the candidates back from dict to list, and copy the updated values
        self.candidates = []
        for cand in self.text_to_candidate.values():
            cand.logp_blank = cand.new_logp_blank
            cand.logp_non_blank = cand.new_logp_non_blank
            self.candidates.append(cand)
        self.text_to_candidate = None

        self._prune_candidates()

    def finalize_scores(self):
        """
        This essentially adds a word space at the end to score the last word.
        The resulting score is in candidates[...].logp_blank, the higher the better.
        """
        if self.candidates_finalized:
            return
        self.candidates_finalized = True
        for cand in self.candidates:
            new_logp_blank = cand.logp_total()
            last_word = cand.text_state.last_word
            if self.lm is not None  and  last_word != '':
                # Merging cands with texts differing only in the final sep was not done in the reference.
                new_lm_state = kenlm.State()
                logp_lm_last_word = self.lm.BaseScore(cand.lm_state, last_word, new_lm_state) * self.log10_to_ln
                cand.lm_state = new_lm_state
                if self.oov_score is not None  and  last_word not in self.lm:
                    logp_lm_last_word = self.oov_score
                new_logp_blank += self.alpha * logp_lm_last_word + self.beta
            cand.logp_blank = new_logp_blank
            cand.logp_non_blank = -np.inf
            cand.new_logp_blank = None
            cand.new_logp_non_blank = None

    def get_top_transcript(self):
        if not self.candidates_finalized:
            self.finalize_scores()
        self._prune_candidates(1)
        return self.candidates[0].text_state.text

    def _get_extended_candidate(self, old_cand, new_char, new_char_index):
        """
        Find existing or create new CtcBeamSearchCandidate with text
        extended with the given character.
        """
        new_text_state, new_word = old_cand.text_state.extended(new_char, new_char_index, sep=self.sep)
        if self.allowed_prefixes is not None and (new_word or new_text_state.last_word) not in self.allowed_prefixes:
            return None, None
        new_cand = self.text_to_candidate.get(new_text_state.text, None)
        if new_cand is None:
            new_cand = CtcBeamSearchCandidate(old_cand)
            self.text_to_candidate[new_text_state.text] = new_cand
            new_cand.text_state = new_text_state
            new_cand.new_logp_blank = -np.inf
            new_cand.new_logp_non_blank = -np.inf
        return new_cand, new_word

    def _prune_candidates(self, beam_width=None):
        """
        Keep top beam_width candidates by score.  Candidates are taken from
        self.candidates, and updated in place.  The score is computed from
        cand.logp_blank and cand.logp_non_blank.  Candidates are not sorted.

            Side effects:
        self.candidates is updated
        """
        if beam_width is None:
            beam_width = self.beam_width
        if len(self.candidates) <= beam_width:
            return
        neg_scores = np.array([-cand.logp_total() for cand in self.candidates])
        parted_indices = np.argpartition(neg_scores, beam_width - 1)
        self.candidates = np.array(self.candidates)[parted_indices[:beam_width]].tolist()


class CtcBeamSearchCandidate:
    def __init__(self, other=None):
        """
        Without args: make an uninitialized CtcBeamSearchCandidate.
        With args: shallow copy words and lm_state to a new CtcBeamSearchCandidate,
          set logp_* fields to -np.inf.
        """
        if other is None:
            return
        self.text_state = other.text_state
        self.lm_state = other.lm_state
        # Log-probability that text ends in blank before removing duplicates and blanks.
        # "logp" here stands for ln(probability), where ln() is natural logarithm.
        self.logp_blank = -np.inf
        # Log-probability that text ends in a non-blank before removing duplicates and blanks
        self.logp_non_blank = -np.inf
        # The .new_logp_* fields will temporarily store the updated values during the update loop
        self.new_logp_blank = None
        self.new_logp_non_blank = None

    @staticmethod
    def empty(lm=None, start_with_bos=True):
        """
        Create an empty initial candidate

            Args:
        with_lm (bool), if LM state should be used
        start_with_bos (bool), True = initialize LM state with start-of-sentence
          False = initialize LM state with empty context (anywhere in the sentence)
        """
        self = CtcBeamSearchCandidate()
        # State of the candidate text after removing duplicates and blanks
        self.text_state = TextState.empty()
        self.logp_blank = 0.
        self.logp_non_blank = -np.inf
        if lm is not None:
            # self.lm_state relates to all words except the last unfinished word
            self.lm_state = kenlm.State()
            if start_with_bos:
                lm.BeginSentenceWrite(self.lm_state)
            else:
                lm.NullContextWrite(self.lm_state)
        else:
            self.lm_state = None
        return self

    def logp_total(self):
        return log_sum_exp(self.logp_blank, self.logp_non_blank)


class DumbDecoder(Adapter):
    __provider__ = 'dumb_decoder'
    prediction_types = (CharacterRecognitionPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'alphabet': ListField(optional=True, default=None, value_type=str, allow_empty=False,
                                  description="Alphabet as list of strings."),
            'uppercase': BoolField(optional=True, default=True, description="Transform result to uppercase"),

        })
        return parameters

    def configure(self):
        self.alphabet = self.get_value_from_config('alphabet') or ' ' + string.ascii_lowercase + '\''
        self.alphabet = self.alphabet.encode('ascii').decode('utf-8')
        self.uppercase = self.get_value_from_config('uppercase')

    def process(self, raw, identifiers=None, frame_meta=None):
        assert len(identifiers) == 1
        decoded = ''.join(self.alphabet[t] for t in raw[0])
        if self.uppercase:
            decoded = decoded.upper()
        return [CharacterRecognitionPrediction(identifiers[0], decoded.upper())]


class TextState:
    __slots__ = ('text', 'last_word', 'last_char_index')

    def __init__(self, text, last_word, last_char_index):
        self.text = text
        self.last_word = last_word  # empty after accepting a word sep
        self.last_char_index = last_char_index

    @staticmethod
    def empty():
        return TextState('', '', None)

    def extended(self, new_char, new_char_index, sep=' '):
        """
        Return a copy of self extended with a new character.  Use sep='' for
        character-based LMs.

            Return pair:
        TextState, new state with the string value extended by new_char
        str or None, the new word if word space encountered, or None otherwise
        """
        if new_char == sep:
            return TextState(self.text + new_char, '', new_char_index), self.last_word
        if sep == '':
            return TextState(self.text + new_char, new_char, new_char_index), self.last_word
        return TextState(self.text + new_char, self.last_word + new_char, new_char_index), None


def log_sum_exp(a, b):
    """
    Logically equivalent to:
      return math.log(math.exp(a) + math.exp(b))
    but avoids the possible over/underflow in math.exp().
    """
    # Checking for -inf is here to only silence runtime warning.
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a

    max_a_b = max(a, b)
    diff_a_b = min(abs(a - b), 35)
    return max_a_b + math.log(math.exp(-diff_a_b) + 1)


def read_vocabulary_prefixes(lm_filename, vocab_offset, vocab_length):
    """
    Extract the set of all possible prefixes of vocabulary words from LM file
    in kenlm binary format.

        Args:
    lm_filename (pathlib.Path)
    vocab_offset (int)
    vocab_length (int or None), None defaults to spanning until the end of file

        Return:
    set of str with all possible prefixes of the words in the vocabulary.
    """
    if vocab_length is None:
        vocab_length = lm_filename.stat().st_size - vocab_offset
        if vocab_length <= 0:
            raise RuntimeError("lm_vocabulary_offset parameter beyond the end of file.")
    elif vocab_offset + vocab_length > lm_filename.stat().st_size:
        raise RuntimeError("lm_vocabulary_offset + lm_vocabulary_length beyond the end of file.")

    with open(str(lm_filename), 'rb') as lm_file:
        lm_file.seek(vocab_offset)
        vocab_data = lm_file.read(vocab_length)

    if len(vocab_data) < 6 or vocab_data[:6] != b'<unk>\0':
        raise RuntimeError(
            "LM vocabulary section does not start with \"<unk>\\0\".  Wrong value of lm_vocabulary_offset parameter?  "
            "lm_vocabulary_offset should point to \"<unk>\" in lm_file."
        )
    if vocab_data[-1:] != b'\0':
        raise RuntimeError(
            "The last byte is LM vocabulary strings section is not 0.  Wrong value of lm_vocabulary_length parameter?  "
            "Omitting this parameter results in vocabulary strings section spanning to the end of file."
        )

    vocab_list = vocab_data[:-1].decode('utf8').split('\0')

    def all_prefixes_word(word):
        # Skipping the empty prefix
        for prefix_len in range(1, len(word)+1):
            yield word[:prefix_len]

    def all_prefixes_vocab(vocab_list):
        for word in vocab_list:
            for prefix in all_prefixes_word(word):
                yield prefix
        yield ''

    return set(all_prefixes_vocab(vocab_list))
