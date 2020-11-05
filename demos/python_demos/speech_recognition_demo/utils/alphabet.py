#
# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""
Manage alphabets.

Alphabets are stored in character lists as list(str).
"""

import codecs


def load_alphabet(filename):
    characters = []
    with codecs.open(filename, 'r', 'utf-8') as f:
        for line in f:
            line = line.rstrip('\r\n')
            if line == '':  # empty line ends the alphabet
                break
            if line[0] == '#':  # comment
                continue
            if line.startswith('\\s'):  # "\s" for space as the first character
                line = ' ' + line[2:]
            elif line[0] == '\\':  # escaping, to enter "#" or "\" as the first character
                line = line[1:]
            characters.append(line)
    return characters


def get_default_alphabet():
    default_alphabet_characters = [
        ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
        'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
        'x', 'y', 'z', '\'',
    ]
    return default_alphabet_characters


class CtcdecoderAlphabet:
    def __init__(self, characters):
        """
        CtcdecoderAlphabet is an object holding alphabet for beam search CTC decoder.

          Args:
        characters (list(str)), lists alphabet characters. Char numeric id
            is 0-based index into this list.
        """
        self.characters = characters

    def decode(self, keys):
        return ''.join(self.characters[key] for key in keys)
