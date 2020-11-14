#! /usr/bin/env python3
#
# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import argparse

from mds_convert_utils import scorer_cut_trie_v6, trie_v6_extract_vocabulary, kenlm_v5_insert_vocabulary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Mozilla DeepSpeech LM scorer from v0.7.x/0.8.x format to v0.6.1 format")
    parser.add_argument('input', type=str, help="Input filename")
    parser.add_argument('output', type=str, help="Output filename")
    parser.add_argument('--trie-offset', type=int, default=None, help="TRIE section offset (optional)")
    parser.add_argument('--no-drop-space', action='store_true',
                        help="Don't remove space at the end of each vocabulary word")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite output file if exists")
    parser.add_argument('--silent', action='store_true', help="Be silent if everything is OK")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.input, 'rb') as f:
        data_scorer = f.read()

    data_scorer, data_trie, trie_offset = scorer_cut_trie_v6(data_scorer, trie_offset=args.trie_offset)
    vocabulary, metadata = trie_v6_extract_vocabulary(data_trie, base_offset=trie_offset)
    data_scorer, vocab_offset = kenlm_v5_insert_vocabulary(data_scorer, vocabulary,
                                                           drop_final_spaces=not args.no_drop_space)

    with open(args.output, 'xb' if not args.overwrite else 'wb') as f:
        f.write(data_scorer)

    if not args.silent:
        # pylint: disable=bad-function-call
        print('# Language model parameters:')
        print('lm_alpha:', metadata['alpha'])
        print('lm_beta:', metadata['beta'])
        if vocab_offset is not None:
            print('lm_vocabulary_offset:', vocab_offset)


if __name__ == '__main__':
    main()
