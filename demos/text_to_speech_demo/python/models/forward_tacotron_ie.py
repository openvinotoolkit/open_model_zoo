"""
 Copyright (c) 2020 Intel Corporation

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

import os.path as osp

import numpy as np

from utils.text_preprocessing import text_to_sequence, _symbol_to_id


class ForwardTacotronIE:
    def __init__(self, model_duration, model_forward, ie, device='CPU', verbose=False):
        self.verbose = verbose
        self.device = device

        self.ie = ie

        self.duration_predictor_net = self.load_network(model_duration)
        self.duration_predictor_exec = self.create_exec_network(self.duration_predictor_net)

        self.forward_net = self.load_network(model_forward)
        self.forward_exec = self.create_exec_network(self.forward_net)

        # fixed length of the sequence of symbols
        self.duration_len = self.duration_predictor_net.input_info['input_seq'].input_data.shape[1]
        # fixed length of the input embeddings for forward
        self.forward_len = self.forward_net.input_info['data'].input_data.shape[1]
        if self.verbose:
            print('Forward limitations : {0} symbols and {1} embeddings'.format(self.duration_len, self.forward_len))
        self.is_attention = 'pos_mask' in self.forward_net.input_info
        if self.is_attention:
            self.init_pos_mask()
            print("Load ForwardTacotron with attention")
        else:
            self.pos_mask = None

    def init_pos_mask(self, mask_sz=6000, window_size=4):
        mask_arr = np.zeros((1, 1, mask_sz, mask_sz), dtype=np.float32)
        width = 2 * window_size + 1
        for i in range(mask_sz - width):
            mask_arr[0][0][i][i:i + width] = 1.0

        self.pos_mask = mask_arr

    @staticmethod
    def sequence_mask(length, max_length=None):
        if max_length is None:
            max_length = np.max(length)
        x = np.arange(max_length, dtype=length.dtype)
        x = np.expand_dims(x, axis=(0))
        length = np.expand_dims(length, axis=(1))
        return x < length

    def seq_to_indexes(self, text):
        res = text_to_sequence(text)
        if self.verbose:
            print(res)
        return res

    @staticmethod
    def build_index(duration, x):
        duration[np.where(duration < 0)] = 0
        tot_duration = np.cumsum(duration, 1)
        max_duration = int(tot_duration.max().item())
        index = np.zeros([x.shape[0], max_duration, x.shape[2]], dtype='long')

        for i in range(tot_duration.shape[0]):
            pos = 0
            for j in range(tot_duration.shape[1]):
                pos1 = tot_duration[i, j]
                index[i, pos:pos1, :] = j
                pos = pos1
            index[i, pos:, :] = j
        return index

    @staticmethod
    def gather(a, dim, index):
        expanded_index = [index if dim == i else np.arange(a.shape[i]).reshape(
                                                  [-1 if i == j else 1 for j in range(a.ndim)]) for i in range(a.ndim)]
        return a[tuple(expanded_index)]

    def load_network(self, model_xml):
        model_bin_name = ".".join(osp.basename(model_xml).split('.')[:-1]) + ".bin"
        model_bin = osp.join(osp.dirname(model_xml), model_bin_name)
        print("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
        net = self.ie.read_network(model=model_xml, weights=model_bin)
        return net

    def create_exec_network(self, net):
        exec_net = self.ie.load_network(network=net, device_name=self.device)
        return exec_net

    def infer_duration(self, sequence, alpha=1.0, non_empty_symbols=None):
        if self.is_attention:
            input_mask = self.sequence_mask(np.array([[non_empty_symbols]]), sequence.shape[1])
            pos_mask = self.pos_mask[:, :, :sequence.shape[1], :sequence.shape[1]]
            out = self.duration_predictor_exec.infer(inputs={"input_seq": sequence,
                                                             "input_mask": input_mask,
                                                             "pos_mask": pos_mask})
        else:
            out = self.duration_predictor_exec.infer(inputs={"input_seq": sequence})
        duration = out["duration"] * alpha

        duration = (duration + 0.5).astype('int').flatten()
        duration = np.expand_dims(duration, axis=0)
        preprocessed_embeddings = out["embeddings"]

        if non_empty_symbols is not None:
            duration = duration[:, :non_empty_symbols]
            preprocessed_embeddings = preprocessed_embeddings[:, :non_empty_symbols]
        indexes = self.build_index(duration, preprocessed_embeddings)
        if self.verbose:
            print("Index: {0}, duration: {1}, embeddings: {2}, non_empty_symbols: {3}"
                  .format(indexes.shape, duration.shape, preprocessed_embeddings.shape, non_empty_symbols))

        return self.gather(preprocessed_embeddings, 1, indexes)

    def infer_mel(self, aligned_emb, non_empty_symbols):
        if self.is_attention:
            data_mask = self.sequence_mask(np.array([[non_empty_symbols]]), aligned_emb.shape[1])
            pos_mask = self.pos_mask[:, :, :aligned_emb.shape[1], :aligned_emb.shape[1]]
            out = self.forward_exec.infer(inputs={"data": aligned_emb,
                                                  "data_mask": data_mask,
                                                  "pos_mask": pos_mask})
        else:
            out = self.forward_exec.infer(inputs={"data": aligned_emb})
        return out['mel'][:, :non_empty_symbols]

    def find_optimal_delimiters_position(self, sequence, delimiters, idx, window=20):
        res = {d: -1 for d in delimiters}
        for i in range(max(0, idx - window), idx):
            if sequence[i] in delimiters:
                res[sequence[i]] = i + 1
        return res

    def forward_duration_prediction_by_delimiters(self, text, alpha):
        sequence = self.seq_to_indexes(text)
        seq_len = len(sequence)
        outputs = []

        if seq_len <= self.duration_len:
            non_empty_symbols = len(sequence) + min(1, self.duration_len - seq_len)
            sequence = sequence + [_symbol_to_id[' ']] * (self.duration_len - seq_len)
            sequence = np.array(sequence)
            sequence = np.expand_dims(sequence, axis=0)
            outputs.append(self.infer_duration(sequence, alpha, non_empty_symbols=non_empty_symbols))
        else:
            punctuation = '.!?,;: '
            delimiters = [_symbol_to_id[p] for p in punctuation]

            start_idx = 0
            while start_idx < seq_len:
                if start_idx + self.duration_len < seq_len:
                    positions = self.find_optimal_delimiters_position(sequence, delimiters,
                                                                      start_idx + self.duration_len,
                                                                      window=self.duration_len//10)
                else:
                    positions = {delimiters[0]: seq_len}
                edge = -1
                for d in delimiters:
                    if positions[d] > 0:
                        edge = positions[d]
                        break
                if edge < 0:
                    raise Exception("Bad delimiter position {0} for sequence with length {1}".format(edge, seq_len))

                sub_sequence = sequence[start_idx:edge]
                non_empty_symbols = len(sub_sequence) + min(1, self.duration_len - len(sub_sequence))
                sub_sequence += [_symbol_to_id[' ']] * (self.duration_len - len(sub_sequence))
                sub_sequence = np.array(sub_sequence)
                sub_sequence = np.expand_dims(sub_sequence, axis=0)
                outputs.append(self.infer_duration(sub_sequence, alpha, non_empty_symbols=non_empty_symbols))
                start_idx = edge

        aligned_emb = np.concatenate(outputs, axis=1)
        return aligned_emb

    def forward(self, text, alpha=1.0):
        aligned_emb = self.forward_duration_prediction_by_delimiters(text, alpha)

        mels = []
        start_idx = 0
        end_idx = 0
        while start_idx < aligned_emb.shape[1] and end_idx < aligned_emb.shape[1]:
            end_idx = min(start_idx + self.forward_len, aligned_emb.shape[1])
            sub_aligned_emb = aligned_emb[:, start_idx:end_idx, :]
            if sub_aligned_emb.shape[1] < self.forward_len:
                sub_aligned_emb = np.pad(sub_aligned_emb,
                                         ((0, 0), (0, self.forward_len - sub_aligned_emb.shape[1]), (0, 0)),
                                         'constant', constant_values=0)
            if self.verbose:
                print("SAEmb shape: {0}".format(sub_aligned_emb.shape))
            mel = self.infer_mel(sub_aligned_emb, end_idx - start_idx)
            mels.append(mel)
            start_idx += self.forward_len

        res = np.concatenate(mels, axis=1)
        if self.verbose:
            print("MEL shape :{0}".format(res.shape))

        return res
