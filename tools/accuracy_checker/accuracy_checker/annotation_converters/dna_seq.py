"""
Copyright (c) 2018-2021 Intel Corporation

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

import numpy as np
from .format_converter import ConverterReturn, BaseFormatConverter
from ..config import PathField, NumberField, ListField
from ..representation import DNASequenceAnnotation


class DNASequenceDatasetConverter(BaseFormatConverter):
    __provider__ = 'dna_sequence'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'num_chunks': NumberField(optional=True, value_type=int, min_value=1),
            'chunks_file': PathField(),
            'ref_file': PathField(),
            'alphabet': ListField(optional=True, default=["N", "A", "C", "G", "T"])
        })
        return params

    def configure(self):
        self.num_chunks = self.get_value_from_config('num_chunks')
        self.chunk_file = self.get_value_from_config('chunks_file')
        self.ref_file = self.get_value_from_config('ref_file')
        self.alphabet = self.get_value_from_config('alphabet')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        chunks = np.load(self.chunk_file)
        references = np.load(self.ref_file)
        if self.num_chunks and chunks.shape[0] > self.num_chunks:
            chunks = chunks[:self.num_chunks]
            references = references[:self.num_chunks]
        num_iterations = chunks.shape[0]
        annotations = []
        for idx, (chunk, ref) in enumerate(zip(chunks, references)):
            annotations.append(DNASequenceAnnotation('chunk_{}'.format(idx), self.decode_ref(ref), chunk))
            if progress_callback and idx % progress_interval == 0:
                progress_callback(idx * 100 / num_iterations)

        return ConverterReturn(annotations, {'label_map': dict(enumerate(self.alphabet))}, None)

    def decode_ref(self, ref):
        return ''.join(self.alphabet[e] for e in ref if e)
