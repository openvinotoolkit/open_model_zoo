"""
Copyright (c)  2018-2021 Intel Corporation

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

import multiprocessing
import unicodedata
import string

import numpy as np

from ..representation import QuestionAnsweringEmbeddingAnnotation
from ..utils import read_json
from ..config import PathField, NumberField, BoolField

from .format_converter import BaseFormatConverter, ConverterReturn


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

def encode_squad_article(article, vocab, do_lower_case):
    def encode_txt(txt):
        if do_lower_case:
            txt = txt.lower()
        return text_to_tokens(txt, vocab)

    for par in article['paragraphs']:
        par['context_enc'], par['context_enc_pos'] = encode_txt(par['context'])
        for qa in par['qas']:
            qa['question_enc'], qa['question_enc_pos'] = encode_txt(qa['question'])

    return article

class SQUADConverterEMB(BaseFormatConverter):
    __provider__ = "squad_emb"
    annotation_types = (QuestionAnsweringEmbeddingAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'testing_file': PathField(description="Path to testing file."),
            'vocab_file': PathField(description='Path to vocabulary file.'),
            'max_seq_length': NumberField(
                description='The maximum total input sequence length after WordPiece tokenization.',
                optional=True, default=128, value_type=int
            ),
            'max_query_length': NumberField(
                description='The maximum number of tokens for the question.',
                optional=True, default=64, value_type=int
            ),
            'lower_case': BoolField(optional=True, default=False, description='Switch tokens to lower case register')
        })

        return configuration_parameters

    def configure(self):
        self.testing_file = self.get_value_from_config('testing_file')
        self.max_seq_length = int(self.get_value_from_config('max_seq_length'))
        self.max_query_length = self.get_value_from_config('max_query_length')
        self.lower_case = self.get_value_from_config('lower_case')
        vocab_file = str(self.get_value_from_config('vocab_file'))
        with open(vocab_file, "r", encoding="utf-8") as r:
            self.vocab = {t.rstrip("\n"): i for i, t in enumerate(r.readlines())}

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):

        squad = read_json(self.testing_file)

        N = len(squad['data'])
        with multiprocessing.Pool() as pool:
            squad['data'] = pool.starmap(
                encode_squad_article,
                zip(squad['data'], [self.vocab] * N, [self.lower_case] * N)
            )

        pad = [self.vocab["[PAD]"]]
        cls = [self.vocab["[CLS]"]]
        sep = [self.vocab["[SEP]"]]

        index_ref = [0]
        def add_sample(ids, max_len, context_pos_id, annotations):
            ids_len = min(max_len - 2, len(ids))
            ids = ids[:ids_len]
            rest = max_len - (ids_len + 2)
            assert rest >= 0

            annotations.append(QuestionAnsweringEmbeddingAnnotation(
                ['{}_{}'.format(n, index_ref[0]) for n in ('input_ids', 'input_mask', 'segment_ids', 'position_ids')],
                np.array(cls + ids + sep + pad * rest),
                np.array([1] * (1 + ids_len + 1) + pad * rest),
                np.array([0] * (1 + ids_len + 1) + pad * rest),
                np.arange(max_len),
                context_pos_id
            ))
            index_ref[0] += 1

        c_annos = []
        q_annos = []
        for article in squad['data']:
            for par in article['paragraphs']:

                add_sample(
                    par['context_enc'],
                    self.max_seq_length,
                    None,
                    c_annos)

                for qa in par['qas']:
                    add_sample(
                        qa['question_enc'],
                        self.max_query_length,
                        c_annos[-1].identifier,
                        q_annos)

        return ConverterReturn(c_annos+q_annos, None, None)
