#
# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import re
import itertools

from collections import OrderedDict

from .struct_utils import parse_format
from .parse_openfst import parse_openfst


DEFAULT_ALPHABET = [b' abcdefghijklmnopqrstuvwxyz\''[i:i+1] for i in range(28)]


def find_trie_v6_offset(data):
    template = re.compile(b'EIRT.\0\0\0..{8}.{8}\xD6\xFD\xB2\x7E')
    matches = template.finditer(data)
    matches = list(itertools.islice(matches, 2))
    if len(matches) == 0:
        raise ValueError("Cannot find OpenFst TRIE dictionary")
    if len(matches) > 1:
        raise ValueError("Found more than one OpenFst TRIE signature in LM file, cannot decide which one is right")
    return matches[0].start()


def scorer_cut_trie_v6(data, trie_offset=None):
    """
    Cut scorer file from Mozilla DeepSpeech v0.7.x/v0.8.x into kenlm and trie sections.
    """
    if trie_offset is None:
        trie_offset = find_trie_v6_offset(data)
    kenlm_data, trie_data = data[:trie_offset], data[trie_offset:]
    return kenlm_data, trie_data, trie_offset


def trie_v6_extract_vocabulary(data, alphabet=None, base_offset=0, max_num_words=10000000):
    if alphabet is None:
        alphabet = DEFAULT_ALPHABET
    fst, _ = parse_trie_v6(data, pos=0, base_offset=base_offset)
    vocabulary = traverse_fst(fst, alphabet, max_num_words=max_num_words)
    return vocabulary, fst.meta


def parse_trie_v6(data, pos=0, base_offset=0):
    (magic, version, is_utf8_mode, alpha, beta), pos = parse_format('<4si?dd', data, pos)
    if magic != b'EIRT':
        raise ValueError("Not a ds_ctcdecoder TRIE section: wrong section signature")
    if version != 6:
        raise ValueError("Wrong ds_ctcdecoder TRIE section version: version {}, expected version 6".format(version))
    if is_utf8_mode:
        raise ValueError("UTF-8 mode language model: UTF-8 mode was not tested, stopping")
    fst, pos = parse_openfst(data, pos, base_offset=base_offset)
    fst.meta.alpha = alpha
    fst.meta.beta = beta
    return fst, pos


def traverse_fst(fst, alphabet, max_num_words):
    graph = states_arcs_to_arc_dict(fst.states, fst.arcs)
    vocabulary = []

    init_state = fst.meta.start_state
    cur_prefix = []
    def process_words(state, prefix):
        out_arcs = graph[state].items()
        if len(out_arcs) == 0:
            word = b''.join(prefix)
            vocabulary.append(word)
            if len(vocabulary) > max_num_words:
                raise RuntimeError("Number of words in vocabulary exceeds limit ({}), stopping".format(max_num_words))
            return
        prefix.append(None)
        for in_label, next_state in out_arcs:
            prefix[-1] = alphabet[in_label - 1]
            process_words(next_state, prefix)
        prefix.pop()
    process_words(init_state, cur_prefix)
    return vocabulary


def states_arcs_to_arc_dict(states, arcs):
    graph = []  # list(state -> dict(in_label -> state))
    for state_desc in states:
        weight, first_arc, num_arcs, num_in_eps, num_out_eps = state_desc  # pylint: disable=unused-variable
        if num_in_eps != 0 or num_out_eps != 0:
            raise ValueError("epsilon arcs are not allowed")

        out_arcs = OrderedDict()
        for arc_desc in arcs[first_arc : first_arc + num_arcs]:
            in_label, out_label, weight, next_state = arc_desc
            if in_label != out_label:
                raise ValueError("FST format changed: out_label differs from in_label")
            out_arcs[in_label] = next_state
        graph.append(out_arcs)

    return graph


def kenlm_v5_insert_vocabulary(data_kenlm, vocabulary, drop_final_spaces=True):
    kenlm_signature = b'mmap lm http://kheafield.com/code format version 5\n\0'
    with_vocab_offset = 0x64
    num_words_offset = 0x6c

    if not data_kenlm.startswith(kenlm_signature):
        raise ValueError("Wrong signature in kenlm section: either broken file, or unsupported kenlm version")
    with_vocab = bool(data_kenlm[with_vocab_offset])
    (num_words,), _ = parse_format('<Q', data_kenlm[num_words_offset:num_words_offset+8])

    if with_vocab:
        # data_kenlm already contains vocabulary section. Do nothing.
        return data_kenlm, None

    if num_words != len(vocabulary) + 3:
        # "3" here for <unk>, <s> and </s>
        raise ValueError("number of words in kenlm header does not match vocabulary size: {} != {}"
                         .format(num_words, len(vocabulary) + 3))

    vocab_offset = len(data_kenlm)
    data_kenlm = [data_kenlm[:with_vocab_offset], b'\1', data_kenlm[with_vocab_offset + 1:]]
    data_kenlm.append(convert_vocabulary_to_kenlm_format(vocabulary, drop_final_spaces=drop_final_spaces))

    return b''.join(data_kenlm), vocab_offset


def convert_vocabulary_to_kenlm_format(vocabulary, drop_final_spaces=True):
    # Unlike kenlm, we don't sort words in MurMurHash order. Our demo doesn't care about word order.
    data_vocab = [b'<unk>\0<s>\0</s>\0']

    if drop_final_spaces:
        if not all(word.endswith(b' ') for word in vocabulary):
            raise ValueError("Some words in vocabulary don't end with a space")
        data_vocab += [word[:-1] + b'\0' for word in vocabulary]
    else:
        data_vocab += [word + b'\0' for word in vocabulary]

    return b''.join(data_vocab)
