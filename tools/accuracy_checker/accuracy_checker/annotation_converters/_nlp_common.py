import unicodedata
from collections import OrderedDict
try:
    import sentencepiece as spm
except ImportError:
    spm = None
from ..config import ConfigError
from ..utils import contains_all


SPIECE_UNDERLINE = '\N{LOWER ONE EIGHTH BLOCK}'
SEG_ID_A = 0
SEG_ID_B = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4
special_symbols = {
    "<unk>": 0,
    "<s>": 1,
    "</s>": 2,
    "<cls>": 3,
    "<sep>": 4,
    "<pad>": 5,
    "<mask>": 6,
    "<eod>": 7,
    "<eop>": 8,
}

UNK_ID = special_symbols["<unk>"]
CLS_ID = special_symbols["<cls>"]
SEP_ID = special_symbols["<sep>"]
MASK_ID = special_symbols["<mask>"]
EOD_ID = special_symbols["<eod>"]


WORD_PIECE_PARAMETERS = ['vocab_file']
SENTENCE_PIECE_PARAMETERS = ['sentence_piece_model_file']


def get_tokenizer(config, lower_case):
    tokenizer = None
    if contains_all(config, WORD_PIECE_PARAMETERS + SENTENCE_PIECE_PARAMETERS):
        raise ConfigError(
            'tokenization method can not be understood correctly from parameters, please provide: \n'
            'for WordPiece tokenization - {}\nfor SentencePiece tokenization - {}\n'.format(
                ', '.join(WORD_PIECE_PARAMETERS), ', '.join(SENTENCE_PIECE_PARAMETERS))
        )
    if contains_all(config, WORD_PIECE_PARAMETERS):
        tokenizer = WordPieceTokenizer(config['vocab_file'], lower_case)

    if contains_all(config, SENTENCE_PIECE_PARAMETERS):
        tokenizer = SentencePieceTokenizer(config['sentence_piece_model_file'], lower_case)

    if tokenizer is None:
        raise ConfigError(
            'tokenization parameters is not found, please provide: \n'
            'for WordPiece tokenization - {}\nfor SentencePiece tokenization - {}\n'.format(
                ', '.join(WORD_PIECE_PARAMETERS), ', '.join(SENTENCE_PIECE_PARAMETERS))
        )
    return tokenizer


class WordPieceTokenizer:
    def __init__(self, vocab_file, lower_case=True, tokenize_chinese_chars=True, max_len=None):
        self.vocab = self.load_vocab(vocab_file)
        self.ids_to_tokens = OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.lower_case = lower_case
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.max_len = max_len

    @staticmethod
    def _run_strip_accents(text):
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    @staticmethod
    def _run_split_on_punc(text, never_split=None):
        """Splits punctuation on a piece of text."""

        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]


    @staticmethod
    def basic_tokenizer(text, lower_case=True, tokenize_chinese_chars=True, never_split=None):
        never_split = never_split or []
        if isinstance(text, bytes):
            text = text.decode("utf-8", "ignore")
        text = _clean_text(text)

        if tokenize_chinese_chars:
            text = WordPieceTokenizer._tokenize_chinese_chars(text)

        text = text.strip()
        tokens = text.split() if text else []
        split_tokens = []
        for token in tokens:
            if lower_case and token not in never_split:
                token = token.lower()
                token = WordPieceTokenizer._run_strip_accents(token)
            split_tokens.extend(WordPieceTokenizer._run_split_on_punc(token, never_split))

        output_tokens = " ".join(split_tokens)
        output_tokens = output_tokens.strip()
        output_tokens = output_tokens.split() if output_tokens else []
        return output_tokens

    @staticmethod
    def _tokenize_chinese_chars(text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if WordPieceTokenizer._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    @staticmethod
    def _is_chinese_char(cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.

        #pylint:disable=chained-comparison
        #pylint:disable=too-many-boolean-expressions
        if ((0x4E00 <= cp <= 0x9FFF) or  #
                (0x3400 <= cp <= 0x4DBF) or  #
                (0x20000 <= cp <= 0x2A6DF) or  #
                (0x2A700 <= cp <= 0x2B73F) or  #
                (0x2B740 <= cp <= 0x2B81F) or  #
                (0x2B820 <= cp <= 0x2CEAF) or
                (0xF900 <= cp <= 0xFAFF) or  #
                (0x2F800 <= cp <= 0x2FA1F)):  #
            return True

        return False

    def wordpiece_tokenizer(self, text):
        if isinstance(text, bytes):
            text = text.decode("utf-8", "ignore")

        output_tokens = []
        text = text.strip()
        tokens = text.split() if text else []
        for token in tokens:
            chars = list(token)
            if len(chars) > 200:
                output_tokens.append("[UNK]")
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append("[UNK]")
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens

    @property
    def all_special_tokens(self):
        return []

    def tokenize(self, text):
        tokens = []
        for token in self.basic_tokenizer(text, self.lower_case, never_split=self.all_special_tokens):
            for sub_token in self.wordpiece_tokenizer(token):
                tokens.append(sub_token)

        return tokens

    def convert_tokens_to_ids(self, items):
        output = []
        for item in items:
            output.append(self.vocab[item])
        return output

    @staticmethod
    def load_vocab(file):
        vocab = {}
        index = 0
        with open(str(file), 'r', encoding="utf-8") as reader:
            while True:
                token = reader.readline()
                if isinstance(token, bytes):
                    token = token.decode("utf-8", "ignore")
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
        return vocab


class SquadWordPieseTokenizer(WordPieceTokenizer):
    def __init__(self, vocab_file, lower_case=True, tokenize_chinese_chars=True, max_len=None):
        super().__init__(vocab_file, lower_case, tokenize_chinese_chars, max_len)
        self.padding_side = "right"
        self.pad_token_type_id = 0
        self.unk_token = "[UNK]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.cls_token = "[CLS]"
        self.mask_token = "[MASK]"
        self.special_tokens_map = {
            'unk_token': self.unk_token,
            'sep_token': self.sep_token,
            'pad_token': self.pad_token,
            'cls_token': self.cls_token,
            'mask_token': self.mask_token
        }

    def encode(self,
               text,
               text_pair=None,
               add_special_tokens=True,
               max_length=None,
               stride=0,
               truncation_strategy='longest_first',
               pad_to_max_length=False,
               return_tensors=None,
               **kwargs):
        encoded_inputs = self.encode_plus(text,
                                          text_pair=text_pair,
                                          max_length=max_length,
                                          add_special_tokens=add_special_tokens,
                                          stride=stride,
                                          truncation_strategy=truncation_strategy,
                                          pad_to_max_length=pad_to_max_length,
                                          return_tensors=return_tensors,
                                          **kwargs)

        return encoded_inputs["input_ids"]

    def encode_plus(self,
                    text,
                    text_pair=None,
                    add_special_tokens=True,
                    max_length=None,
                    stride=0,
                    truncation_strategy='longest_first',
                    pad_to_max_length=False,
                    return_token_type_ids=True,
                    return_attention_mask=True,
                    return_overflowing_tokens=False,
                    return_special_tokens_mask=False,
                    **kwargs):

        def get_input_ids(text):
            if isinstance(text, str):
                return self.convert_tokens_to_ids(self.tokenize(text))
            if isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                return self.convert_tokens_to_ids(text)
            if isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            raise ValueError(
                "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        first_ids = get_input_ids(text)
        second_ids = get_input_ids(text_pair) if text_pair is not None else None

        return self.prepare_for_model(first_ids,
                                      pair_ids=second_ids,
                                      max_length=max_length,
                                      pad_to_max_length=pad_to_max_length,
                                      add_special_tokens=add_special_tokens,
                                      stride=stride,
                                      truncation_strategy=truncation_strategy,
                                      return_attention_mask=return_attention_mask,
                                      return_token_type_ids=return_token_type_ids,
                                      return_overflowing_tokens=return_overflowing_tokens,
                                      return_special_tokens_mask=return_special_tokens_mask)

    def prepare_for_model(self, ids, pair_ids=None, max_length=None, add_special_tokens=True, stride=0,
                          truncation_strategy='longest_first',
                          pad_to_max_length=False,
                          return_token_type_ids=True,
                          return_attention_mask=True,
                          return_overflowing_tokens=False,
                          return_special_tokens_mask=False):
        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        encoded_inputs = {}

        # Handle max sequence length
        total_len = len_ids + len_pair_ids + (self.num_added_tokens(pair=pair) if add_special_tokens else 0)
        if max_length and total_len > max_length:
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(ids, pair_ids=pair_ids,
                                                                        num_tokens_to_remove=total_len - max_length,
                                                                        truncation_strategy=truncation_strategy,
                                                                        stride=stride)
            if return_overflowing_tokens:
                encoded_inputs["overflowing_tokens"] = overflowing_tokens
                encoded_inputs["num_truncated_tokens"] = total_len - max_length

        # Handle special_tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([1] * len(pair_ids) if pair else [])

        if return_special_tokens_mask:
            encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)

        encoded_inputs["input_ids"] = sequence
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids

        if max_length and len(encoded_inputs["input_ids"]) > max_length:
            encoded_inputs["input_ids"] = encoded_inputs["input_ids"][:max_length]
            if return_token_type_ids:
                encoded_inputs["token_type_ids"] = encoded_inputs["token_type_ids"][:max_length]
            if return_special_tokens_mask:
                encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"][:max_length]

        check_len = max_length and len(encoded_inputs["input_ids"]) < max_length
        check_len2 = max_length is None and len(encoded_inputs["input_ids"]) < self.max_len <= 10000
        needs_to_be_padded = pad_to_max_length and (check_len or check_len2)

        pad_strategy = {'left': pad_left, 'right': pad_right}

        if needs_to_be_padded:
            difference = (max_length if max_length is not None else self.max_len) - len(encoded_inputs["input_ids"])
            encoded_inputs = pad_strategy[self.padding_side](
                encoded_inputs, difference, return_attention_mask, return_token_type_ids, return_special_tokens_mask,
                self.pad_token_id, self.pad_token_type_id
            )

        elif return_attention_mask:
            encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"])

        return encoded_inputs

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def num_added_tokens(self, pair=False):
        token_ids_0 = []
        token_ids_1 = []
        return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))

    @staticmethod
    def truncate_sequences(ids, pair_ids=None, num_tokens_to_remove=0, truncation_strategy='longest_first', stride=0):
        if num_tokens_to_remove <= 0:
            return ids, pair_ids, []

        if truncation_strategy == 'longest_first':
            overflowing_tokens = []
            for _ in range(num_tokens_to_remove):
                if pair_ids is None or len(ids) > len(pair_ids):
                    overflowing_tokens = [ids[-1]] + overflowing_tokens
                    ids = ids[:-1]
                else:
                    pair_ids = pair_ids[:-1]
            window_len = min(len(ids), stride)
            if window_len > 0:
                overflowing_tokens = ids[-window_len:] + overflowing_tokens
        elif truncation_strategy == 'only_first':
            assert len(ids) > num_tokens_to_remove
            window_len = min(len(ids), stride + num_tokens_to_remove)
            overflowing_tokens = ids[-window_len:]
            ids = ids[:-num_tokens_to_remove]
        elif truncation_strategy == 'only_second':
            assert pair_ids is not None and len(pair_ids) > num_tokens_to_remove
            window_len = min(len(pair_ids), stride + num_tokens_to_remove)
            overflowing_tokens = pair_ids[-window_len:]
            pair_ids = pair_ids[:-num_tokens_to_remove]
        elif truncation_strategy == 'do_not_truncate':
            raise ValueError("Input sequence are too long for max_length. Please select a truncation strategy.")
        else:
            raise ValueError(
                "Truncation_strategy should be selected in "
                "['longest_first', 'only_first', 'only_second', 'do_not_truncate']"
            )
        return ids, pair_ids, overflowing_tokens

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError("You should not supply a second sequence if the provided sequence of "
                                 "ids is already formated with special tokens for the model.")
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    @property
    def pad_token_id(self):
        return self.vocab[self.pad_token]

    @property
    def unk_token_id(self):
        return self.vocab[self.unk_token]

    @property
    def sep_token_id(self):
        return self.vocab[self.sep_token]

    @property
    def cls_token_id(self):
        return self.vocab[self.cls_token]

    @property
    def all_special_ids(self):
        """ List the vocabulary indices of the special tokens ('<unk>', '<cls>'...) mapped to
            class attributes (cls_token, unk_token...).
        """
        all_toks = self.all_special_tokens
        all_ids = self.convert_tokens_to_ids(all_toks)
        return all_ids

    @property
    def all_special_tokens(self):
        """ List all the special tokens ('<unk>', '<cls>'...) mapped to class attributes
            (cls_token, unk_token...).
        """
        all_toks = []
        set_attr = self.special_tokens_map
        for attr_value in set_attr.values():
            all_toks = all_toks + (list(attr_value) if isinstance(attr_value, (list, tuple)) else [attr_value])
        all_toks = list(set(all_toks))
        return all_toks

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """ Converts a single index or a sequence of indices (integers) in a token "
            (resp.) a sequence of tokens (str/unicode), using the vocabulary and added tokens.
            Args:
                skip_special_tokens: Don't decode special tokens (self.all_special_tokens). Default: False
        """
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            tokens.append(self._convert_id_to_token(index))
        return tokens


    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def wordpiece_tokenizer(self, text):
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > 100:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


    @property
    def max_len_single_sentence(self):
        return self.max_len - self.num_special_tokens_to_add(pair=False)

    @property
    def max_len_sentences_pair(self):
        return self.max_len - self.num_special_tokens_to_add(pair=True)

    def num_special_tokens_to_add(self, pair=False):
        token_ids_0 = []
        token_ids_1 = []
        return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class SentencePieceTokenizer:
    def __init__(self, tokenizer_model, lower_case=True, remove_space=True):
        if spm is None:
            raise ConfigError('Sentence piece tokenizer required sentencepiece, please install it before usage')
        self.encoder = spm.SentencePieceProcessor()
        self.encoder.Load(str(tokenizer_model))
        self.lower_case = lower_case
        self.remove_space = remove_space

    def preprocess_text(self, inputs):
        if self.remove_space:
            outputs = ' '.join(inputs.strip().split())
        else:
            outputs = inputs

        outputs = outputs.replace("``", '"').replace("''", '"')
        if self.lower_case:
            outputs = outputs.lower()

        return outputs

    def encode_ids(self, text, sample=False):
        pieces = self.encode_pieces(text, sample)
        ids = [self.encoder.PieceToId(piece) for piece in pieces]
        return ids

    def encode_pieces(self, text, sample=False):
        if not sample:
            pieces = self.encoder.EncodeAsPieces(text)
        else:
            pieces = self.encoder.SampleEncodeAsPieces(text, 64, 0.1)
        new_pieces = []
        for piece in pieces:
            if len(piece) > 1 and piece[-1] == ',' and piece[-2].isdigit():
                cur_pieces = self.encoder.EncodeAsPieces(
                    piece[:-1].replace(SPIECE_UNDERLINE, ''))
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)
        return new_pieces

    def tokenize(self, text):
        text = self.preprocess_text(text)
        return self.encode_ids(text)

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

def _clean_text(text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xFFFD or _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char in ["\t", "\n", "\r"]:
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False

def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char in [" ", "\t", "\n", "\r"]:
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False

def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def pad_right(
        encoded_inputs, difference,
        return_attention_mask, return_token_type_ids, return_special_tokens_mask,
        pad_token_id, pad_token_type_id
):
    if return_attention_mask:
        encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"]) + [0] * difference
    if return_token_type_ids:
        encoded_inputs["token_type_ids"] = encoded_inputs["token_type_ids"] + [pad_token_type_id] * difference
    if return_special_tokens_mask:
        encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
    encoded_inputs["input_ids"] = encoded_inputs["input_ids"] + [pad_token_id] * difference
    return encoded_inputs


def pad_left(
        encoded_inputs, difference,
        return_attention_mask, return_token_type_ids, return_special_tokens_mask,
        pad_token_id, pad_token_type_id
):
    if return_attention_mask:
        encoded_inputs["attention_mask"] = [0] * difference + [1] * len(encoded_inputs["input_ids"])
    if return_token_type_ids:
        encoded_inputs["token_type_ids"] = [pad_token_type_id] * difference + encoded_inputs[
            "token_type_ids"]
    if return_special_tokens_mask:
        encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
    encoded_inputs["input_ids"] = [pad_token_id] * difference + encoded_inputs["input_ids"]
    return encoded_inputs
