#
# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from types import SimpleNamespace as namespace

from .struct_utils import parse_format, align_pos


def parse_string(data, pos=0):
    (length,), pos = parse_format('<i', data, pos)
    length = max(0, length)
    value, pos = parse_format('<{}s'.format(length), data, pos)
    return value, pos


def parse_header(data, pos=0):
    (magic,), pos = parse_format('<i', data, pos)
    if magic != 0x7eb2fdd6:
        raise ValueError("Bad OpenFst TRIE: wrong magic number")
    (fst_type,), pos = parse_string(data, pos)
    (arc_type,), pos = parse_string(data, pos)
    (version, flags, properties, start_state, num_states, num_arcs), pos = parse_format('<iiQqqq', data, pos)
    if fst_type != b'const' or arc_type != b'standard' or version != 1:
        raise ValueError("Cannot parse OpenFst TRIE: this version of format is not supported")

    return namespace(
        magic=magic, fst_type=fst_type, arc_type=arc_type, version=version,
        flags=flags, properties=properties,
        start_state=start_state, num_states=num_states, num_arcs=num_arcs,
    ), pos


def parse_state(data, pos=0):
    (weight, first_arc, num_arcs, num_in_eps, num_out_eps), pos = parse_format('<fIIII', data, pos)
    return (weight, first_arc, num_arcs, num_in_eps, num_out_eps), pos


def parse_arc(data, pos=0):
    (in_label, out_label, weight, next_state), pos = parse_format('<iifi', data, pos)
    return (in_label, out_label, weight, next_state), pos


def parse_openfst(data, pos=0, base_offset=0):
    header, pos = parse_header(data, pos)

    pos = align_pos(pos, align=16, base_offset=base_offset)
    states = []
    for state_idx in range(header.num_states):  # pylint: disable=unused-variable
        state, pos = parse_state(data, pos)
        states.append(state)

    pos = align_pos(pos, align=16, base_offset=base_offset)
    arcs = []
    for arc_idx in range(header.num_arcs):  # pylint: disable=unused-variable
        arc, pos = parse_arc(data, pos)
        arcs.append(arc)

    fst = namespace(meta=header, states=states, arcs=arcs)
    return fst, pos
