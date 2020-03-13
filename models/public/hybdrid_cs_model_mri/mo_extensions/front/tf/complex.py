"""
 Copyright (C) 2018-2020 Intel Corporation

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

import logging as log

import numpy as np

from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph
from mo.ops.const import Const
from extensions.ops.elementwise import Add, Pow

# This pattern detects subgraphs
#
#        +---> StridedSlice ---+
# input -+                     + -> Complex -> NextOp
#        +---> StridedSlice ---+
#
# With a limitation that input has number of channels 2 and slices extract real
# and imaginary components correspondingly.
#
# Then we replace it pattern to just Identity edge:
#
# input -> NextOp
class Complex(FrontReplacementSubgraph):
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('strided_slice_real', dict(op='StridedSlice')),
                ('strided_slice_imag', dict(op='StridedSlice')),
                ('complex', dict(op='Complex')),
            ],
            edges=[
                ('strided_slice_real', 'complex', {'in': 0}),
                ('strided_slice_imag', 'complex', {'in': 1}),
            ])

    @staticmethod
    def replace_sub_graph(graph: Graph, match: dict):
        strided_slice_real = match['strided_slice_real']
        strided_slice_imag = match['strided_slice_imag']
        cmp = match['complex']

        if strided_slice_real['pb'].input[0] != strided_slice_imag['pb'].input[0]:
            log.debug('The pattern does not correspond to Complex subgraph. Different inputs')
            return
        # TODO: check StridedSlice inputs

        # TODO create identity edge
        one = Const(graph, {'value': np.float32(1.0)}).create_node()
        identity = Pow(graph, dict(name=cmp.name + '/identity', power=1.0)).create_node([strided_slice_real.in_node(0), one])
        cmp.replace_node(identity)
