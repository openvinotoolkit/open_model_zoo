"""
 Copyright (c) 2020 Intel Corporation

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

import unicodedata
import string


# A class to store context as text, its tokens and embedding vector
class ContextData:
    def __init__(self, tokens_id, tokens_se, context=None, emb=None):
        self.c_tokens_id = tokens_id
        self.c_tokens_se = tokens_se
        self.context = context
        self.emb = emb


class ContextWindow:
    def __init__(self, window_len, tokens_id, tokens_se):
        self.tokens_id = tokens_id
        self.tokens_se = tokens_se
        self.window_len = window_len
        self.stride = self.window_len // 2 # overlap by half
        self.total_len = len(self.tokens_id)
        self.s, self.e = 0, min(self.window_len, self.total_len)

    def move(self):
        self.s = min(self.s + self.stride, self.total_len)
        self.e = min(self.s + self.window_len, self.total_len)

    def is_over(self):
        return self.e - self.s < self.stride

    def get_context_data(self, context=None):
        return ContextData(self.tokens_id[self.s:self.e], self.tokens_se[self.s:self.e],
                           context=context)


# load vocabulary file for encoding
def load_vocab_file(vocab_file_name):
    with open(vocab_file_name, "r", encoding="utf-8") as r:
        return {t.rstrip("\n"): i for i, t in enumerate(r.readlines())}


# split word by vocab items and get tok codes
# iteratively return codes
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
#iteratively return words
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
def text_to_tokens(text, vocab):
    tokens_id = []
    tokens_se = []
    for s, e in split_to_words(text):
        for tok in encode_by_voc(text[s:e], vocab):
            tokens_id.append(tok)
            tokens_se.append((s, e))

    return tokens_id, tokens_se
