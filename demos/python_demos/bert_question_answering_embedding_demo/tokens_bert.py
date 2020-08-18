import unicodedata
import string

# split word by vocab items and get tok codes
# iterativly return codes
def encode_by_voc(w, vocab):
    # remove mark and control chars
    def clean_word(w):
        wo = ""  # accumulator for output word
        for c in unicodedata.normalize("NFD", w):
            c_cat = unicodedata.category(c)
            # remove mark nonspacing code and controls
            if c_cat != "Mn" and c_cat[0] != "C":
                wo += c
        return wo

    w = clean_word(w)

    res = []
    for s0, e0 in split_to_words(w):
        s, e = s0, e0
        tokens = []
        while e > s:
            subword = w[s:e] if s == s0 else "##" + w[s:e]
            if subword in vocab:
                tokens.append(vocab[subword])
                s, e = e, e0
            else:
                e -= 1
        if s < e0:
            tokens = [vocab['[UNK]']]
        res.extend(tokens)
    return res

#split big text into words by spaces
#iterativly return words
def split_to_words(text):
    prev_is_sep = True # mark initial prev as space to start word from 0 char
    for i, c in enumerate(text + " "):
        is_punc = (c in string.punctuation or unicodedata.category(c)[0] == "P")
        cur_is_sep = (c.isspace() or is_punc)
        if prev_is_sep != cur_is_sep:
            if prev_is_sep:
                start = i
            else:
                yield start, i
                del start
        if is_punc:
            yield i, i+1
        prev_is_sep = cur_is_sep

# get big text and return list of token id and start-end positions for each id in original texts
def text_to_tokens(text, vocab_or_tokenizer):
    tokens_id = []
    tokens_se = []
    for s, e in split_to_words(text):
        if hasattr(vocab_or_tokenizer, 'encode'):
            #vocab_or_tokenizer is tokenizer
            toks = vocab_or_tokenizer.encode(text[s:e], add_special_tokens=False)
        else:
            #vocab_or_tokenizer is tokens dictionary
            toks = encode_by_voc(text[s:e], vocab_or_tokenizer)

        for tok in toks:
            tokens_id.append(tok)
            tokens_se.append((s, e))

    return tokens_id, tokens_se
