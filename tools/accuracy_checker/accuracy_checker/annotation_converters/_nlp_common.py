import unicodedata
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
    def __init__(self, vocab_file, lower_case=True, tokenize_chinese_chars=True):
        self.vocab = self.load_vocab(vocab_file)
        self.lower_case = lower_case
        self.tokenize_chinese_chars = tokenize_chinese_chars

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
    def _run_split_on_punc(text):
        def _is_punctuation(char):
            punct = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
            if char in punct:
                return True
            cat = unicodedata.category(char)
            if cat.startswith("P"):
                return True
            return False

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

    def basic_tokenizer(self, text):
        if isinstance(text, bytes):
            text = text.decode("utf-8", "ignore")

        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)

        text = text.strip()
        tokens = text.split() if text else []
        split_tokens = []
        for token in tokens:
            if self.lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = " ".join(split_tokens)
        output_tokens = output_tokens.strip()
        output_tokens = output_tokens.split() if output_tokens else []
        return output_tokens

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
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
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
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

    def tokenize(self, text):
        tokens = []
        for token in self.basic_tokenizer(text):
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
        with open(str(file), 'r') as reader:
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
