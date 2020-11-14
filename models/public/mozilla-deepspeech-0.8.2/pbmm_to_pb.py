#! /usr/bin/env python3
#
# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import argparse

from tensorflow import GraphDef

from mds_convert_utils.memmapped_file_system_pb2 import MemmappedFileSystemDirectory


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert memory-mapped Tensorflow protobuf format into plain Tensorflow protobuf (GraphDef)")
    parser.add_argument('input', type=str, help="Input .pbmm filename")
    parser.add_argument('output', type=str, help="Output .pb filename")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite output file if exists")
    return parser.parse_args()


def bytes_to_uint(data):
    res = 0
    for digit in reversed(data):
        res = (res << 8) + digit
    return res


def load_memmapped_fs(data):
    if isinstance(data, str):
        # "data" is a filename, open it and read the data
        with open(data, 'rb') as f:
            data = f.read()

    data_directory_ofs = bytes_to_uint(data[-8:])
    data_directory = data[data_directory_ofs:-8]

    directory_pb = MemmappedFileSystemDirectory()
    directory_pb.ParseFromString(data_directory)

    return {
        entry.name: data[entry.offset : entry.offset + entry.length]
        for entry in directory_pb.element
    }


def load_graph_def(data):
    graph_def = GraphDef()
    graph_def.ParseFromString(data)
    return graph_def


def serialize_graph_def(graph_def):
    return graph_def.SerializeToString()


def convert_node(node, mmfs):
    """
    node may be changed in place.
    """
    node.op = 'Const'
    # region_name is bytes here, while it is string in MemmappedFileSystemDirectory
    region_name = node.attr['memory_region_name'].s.decode('utf-8')
    dtype = node.attr['dtype'].type
    shape = node.attr['shape'].shape
    del node.attr['memory_region_name']
    del node.attr['shape']
    # keep node.attr['dtype']
    node.attr['value'].tensor.dtype = dtype
    node.attr['value'].tensor.tensor_shape.CopyFrom(shape)
    node.attr['value'].tensor.tensor_content = mmfs[region_name]
    return node


def undo_mmap(graph_def, mmfs):
    nodes = []
    for node in graph_def.node:
        if node.op != 'ImmutableConst':
            nodes.append(node)
            continue
        node = convert_node(node, mmfs)
        nodes.append(node)
    del graph_def.node[:]
    graph_def.node.extend(nodes)
    return graph_def


def main():
    args = parse_args()
    mmfs = load_memmapped_fs(args.input)
    ROOT = 'memmapped_package://.'

    graph_def = load_graph_def(mmfs[ROOT])

    graph_def = undo_mmap(graph_def, mmfs)

    with open(args.output, 'xb' if not args.overwrite else 'wb') as f:
        f.write(serialize_graph_def(graph_def))


if __name__ == '__main__':
    main()
