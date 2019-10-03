import unicodedata

class Tokenizer:
    def __init__(self, vocab_file):
        self.vocab = load_vocab(vocab_file)

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

        text = text.strip()
        tokens = text.split() if text else []
        split_tokens = []
        for token in tokens:
            token = token.lower()
            token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = " ".join(split_tokens)
        output_tokens = output_tokens.strip()
        output_tokens = output_tokens.split() if output_tokens else []
        return output_tokens

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
