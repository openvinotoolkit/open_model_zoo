"""
Copyright (c) 2018-2024 Intel Corporation

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
from  collections import namedtuple
import numpy as np
from .adapter import Adapter
from ..config import NumberField, StringField, ConfigError
from ..representation import DNASequencePrediction
from ..utils import UnsupportedPackage

try:
    from fast_ctc_decode import beam_search, viterbi_search
except ImportError as import_error:
    beam_search = UnsupportedPackage('fast_ctc_decode', import_error.msg)
    viterbi_search = UnsupportedPackage('fast_ctc_decode', import_error.msg)



class DNASeqRecognition(Adapter):
    __provider__ = 'dna_seq_beam_search'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'beam_size': NumberField(optional=True, value_type=int, default=5, min_value=1, description='size of beam'),
            'threshold': NumberField(
                optional=True, min_value=0, default=1e-3, description='threshold ofr valid prediction'
            ),
            'output_blob': StringField(optional=True, description='name of output layer')
        })
        return params

    def configure(self):
        if isinstance(beam_search, UnsupportedPackage):
            beam_search.raise_error(self.__provider__)
        self.beam_size = self.get_value_from_config('beam_size')
        self.threshold = self.get_value_from_config('threshold')
        self.output_blob = self.get_value_from_config('output_blob')
        self.output_verified = False

    def process(self, raw, identifiers, frame_meta):
        if not self.label_map:
            raise ConfigError('Beam Search Decoder requires dataset label map for correct decoding.')
        alphabet = list(self.label_map.values())
        raw_outputs = self._extract_predictions(raw, frame_meta)
        if not self.output_verified:
            self.select_output_blob(raw_outputs)
        result = []
        for identifier, out in zip(identifiers, np.exp(raw_outputs[self.output_blob])):
            if self.beam_size == 1:
                seq, _ = viterbi_search(np.squeeze(out), alphabet, False, 1, 0)
            else:
                seq, _ = beam_search(np.squeeze(out.astype(np.float32)), alphabet, self.beam_size, self.threshold)
            result.append(DNASequencePrediction(identifier, seq))
        return result

    def select_output_blob(self, outputs):
        self.output_verified = True
        if self.output_blob:
            self.output_blob = self.check_output_name(self.output_blob, outputs)
            return
        super().select_output_blob(outputs)
        return

class DNASequenceWithCRFAdapter(Adapter):
    __provider__ = 'dna_seq_crf_beam_search'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'output_blob': StringField(optional=True, description='name of output layer')
        })
        return params

    def configure(self):
        try:
            import torch  # pylint: disable=import-outside-toplevel
            self._torch = torch
        except ImportError as torch_import_error:
            UnsupportedPackage('torch', torch_import_error.msg).raise_error(self.__provider__)
        try:
            from crf_beam import beam_search as crf_beam_search  # pylint: disable=import-outside-toplevel
            self._crf_beam_search = crf_beam_search
        except ImportError as crf_beam_import_error:
            UnsupportedPackage('crf_beam', crf_beam_import_error.msg).raise_error(self.__provider__)
        if not self.label_map:
            raise ConfigError('Beam Search Decoder requires dataset label map for correct decoding.')
        alphabet = list(self.label_map.values())
        self.output_blob = self.get_value_from_config('output_blob')
        self.output_verified = False
        self.state_len = len(alphabet)
        self.n_base = self.state_len - 1
        semiring = namedtuple('semiring', ('zero', 'one', 'mul', 'sum', 'dsum'))
        self.log_semiring = semiring(zero=-1e38, one=0., mul=self._torch.add, sum=self._torch.logsumexp,
            dsum=self._torch.softmax)
        self.idx = self._torch.cat([
            self._torch.arange(self.n_base ** (self.state_len))[:, None],
            self._torch.arange(
                self.n_base ** (self.state_len)
            ).repeat_interleave(self.n_base).reshape(self.n_base, -1).T
        ], dim=1).to(self._torch.int32)

    def process(self, raw, identifiers, frame_meta):
        raw_outputs = self._extract_predictions(raw, frame_meta)
        if not self.output_verified:
            self.select_output_blob(raw_outputs)
        result = []
        scores =  self._torch.from_numpy(raw_outputs[self.output_blob])
        fwd_scores = self.forward_scores(scores)
        bwd_scores = self.backward_scores(scores)
        posts = self._torch.softmax(fwd_scores + bwd_scores, dim=-1)
        scores = scores.transpose(0, 1)
        bwds = bwd_scores.transpose(0, 1)
        posts = posts.transpose(0, 1)
        for identifier, score,  bwd, post in zip(identifiers, scores, bwds, posts):
            seq, _, _ = self._crf_beam_search(score, bwd, post)
            result.append(DNASequencePrediction(identifier, seq))
        return result

    @staticmethod
    def scan(ms, idx, v0, s):
        t, n, c, _ = ms.shape
        alpha = ms.new_full((t + 1, n, c), s.zero)
        alpha[0] = v0
        for t_ in range(t):
            alpha[t_ + 1] = s.sum(s.mul(ms[t_], alpha[t_, :, idx]), dim=-1)
        return alpha

    def forward_scores(self, scores):
        t, n, _ = scores.shape
        ms = scores.reshape(t, n, -1, self.n_base + 1)
        v0 = ms.new_full((n, self.n_base ** (self.state_len)), self.log_semiring.one)
        return self.scan(ms, self.idx.to(self._torch.int64), v0, self.log_semiring)

    def backward_scores(self, scores):
        _, n, _ = scores.shape
        vt = scores.new_full((n, self.n_base**(self.state_len)), self.log_semiring.one)
        idx_t = self.idx.flatten().argsort().reshape(*self.idx.shape)
        ms_t = scores[:, :, idx_t]
        idx_t = self._torch.div(idx_t, self.n_base + 1, rounding_mode='floor')
        return self.scan(ms_t.flip(0), idx_t.to(self._torch.int64), vt, self.log_semiring).flip(0)

    def select_output_blob(self, outputs):
        self.output_verified = True
        if self.output_blob:
            self.output_blob = self.check_output_name(self.output_blob, outputs)
            return
        super().select_output_blob(outputs)
        return
