#
# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import struct


__all__ = ['parse_format', 'align_pos']


def parse_format(format_, data, pos=0, align=None, pos_base_offset=0):
    len_ = struct.calcsize(format_)
    return struct.unpack(format_, data[pos : pos + len_]), pos + struct.calcsize(format_)


def align_pos(pos, align, base_offset=0):
    pos += (-(pos + base_offset)) % align
    return pos
