import os.path as osp
import logging as log

import numpy as np
from openvino.inference_engine import IECore

from utils.text_preprocessing import text_to_sequence, _symbol_to_id

class ForwardTacotronIE:
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
        expanded_index = [index if dim==i else np.arange(a.shape[i]).reshape([-1 if i==j else 1 for j in range(a.ndim)]) for i in range(a.ndim)]
        return a[tuple(expanded_index)]

    @staticmethod
    def gather_numpy(arr, dim, index):
        print(arr.shape)
        print(index.shape)
        """
        Gathers values along an axis specified by dim.
        For a 3-D tensor the output is specified by:
            out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
            out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
            out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

        :param dim: The axis along which to index
        :param index: A tensor of indices of elements to gather
        :return: tensor of gathered values
        """
        idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
        self_xsection_shape = arr.shape[:dim] + arr.shape[dim + 1:]
        if idx_xsection_shape != self_xsection_shape:
            raise ValueError("Except for dimension " + str(dim) +
                             ", all dimensions of index and self should be the same size")
        if index.dtype != np.dtype('int_'):
            raise TypeError("The values of index must be integers")
        data_swaped = np.swapaxes(arr, 0, dim)
        index_swaped = np.swapaxes(index, 0, dim)
        gathered = np.choose(index_swaped, data_swaped)
        return np.swapaxes(gathered, 0, dim)

    def load_network(self, model_xml):
        model_bin_name = ".".join(osp.basename(model_xml).split('.')[:-1]) + ".bin"
        model_bin = osp.join(osp.dirname(model_xml), model_bin_name)

        # Read IR
        log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
        net = self.ie.read_network(model=model_xml, weights=model_bin)

        print("#################################################################")
        print("Model: {0}. Inputs: {1}".format(model_xml, net.inputs))
        print("#################################################################")
        print("Model: {0}. Outputs: {1}".format(model_xml, net.outputs))
        exec_net = self.ie.load_network(network=net, device_name=self.device)
        return net, exec_net

    def __init__(self, model_duration, model_forward, device='CPU', verbose=False):
        super().__init__()
        self.verbose = verbose
        self.device = device

        self.ie = IECore()

        self.duration_predictor_net, self.duration_predictor_exec = self.load_network(model_duration)
        self.forward_net, self.forward_exec = self.load_network(model_forward)

        # fixed length of the sequence of symbols
        self.duration_len = self.duration_predictor_net.inputs['input_seq'].shape[1]
        # fixed length of the input embeddings for forward
        self.forward_len = self.forward_net.inputs['data'].shape[1]
        if self.verbose:
            print('Forward limitations : {0} symbols and {1} embeddings'.format(self.duration_len, self.forward_len))

    def forward_duration(self, sequence, alpha=1.0, non_empty_symbols=None):
        out = self.duration_predictor_exec.infer(inputs={"input_seq": sequence})
        duration = out["duration"] * alpha

        duration = (duration + 0.5).astype('int').flatten()
        duration = np.expand_dims(duration, axis=0)
        preprocessed_embeddings = out["embeddings"]

        if non_empty_symbols is not None:
            duration = duration[:, :non_empty_symbols]
            preprocessed_embeddings = preprocessed_embeddings[:,:non_empty_symbols]
        indexes = self.build_index(duration, preprocessed_embeddings)
        if self.verbose:
            print("Index: {0}, duration: {1}, embeddings: {2}, non_empty_symbols: {3}".format(indexes.shape, duration.shape, preprocessed_embeddings.shape, non_empty_symbols))
        return self.gather(preprocessed_embeddings, 1, indexes)

    def forward_forward(self, aligned_emb):
        out = self.forward_exec.infer(inputs={"data": aligned_emb})
        return out['mel']

    def forward(self, text, alpha=1.0):
        sequence = self.seq_to_indexes(text)
        if len(sequence) <= self.duration_len:
            non_empty_symbols = None
            if len(sequence) < self.duration_len:
                non_empty_symbols = len(sequence)
                sequence += [_symbol_to_id[' ']] * (self.duration_len - len(sequence))
            sequence = np.array(sequence)
            sequence = np.expand_dims(sequence, axis=0)
            if self.verbose:
                print("Seq shape: {0}".format(sequence.shape))
            aligned_emb = self.forward_duration(sequence, alpha, non_empty_symbols=non_empty_symbols)
            if self.verbose:
                print("AEmb shape: {0}".format(aligned_emb.shape))
        else:
            punctuation = '!\'(),.:;? '
            delimiters = [_symbol_to_id[p] for p in punctuation]
            # try to found optimal fragmentation for inference
            ranges = [i+1 for i,val in enumerate(sequence) if val in delimiters]
            if len(sequence) not in ranges:
                ranges.append(len(sequence))
            optimal_ranges = []
            prev_begin = 0
            for i in range(len(ranges)-1):
                if ranges[i] < 0:
                    continue
                res1 = (ranges[i] - prev_begin) % self.duration_len
                res2 = (ranges[i + 1] - prev_begin) % self.duration_len
                if res1 > res2 or res1 == 0:
                    if res2 == 0:
                        optimal_ranges.append(ranges[i+1])
                        ranges[i+1] = -1
                    else:
                        optimal_ranges.append(ranges[i])
                    prev_begin = optimal_ranges[-1]
            if self.verbose:
                print(optimal_ranges)
            if len(sequence) not in optimal_ranges:
                optimal_ranges.append(len(sequence))

            outputs = []
            start_idx = 0
            for edge in optimal_ranges:
                sub_sequence = sequence[start_idx:edge]
                start_idx = edge
                non_empty_symbols = None
                if len(sub_sequence) < self.duration_len:
                    non_empty_symbols = len(sub_sequence)
                    sub_sequence += [_symbol_to_id[' ']] * (self.duration_len - len(sub_sequence))
                sub_sequence = np.array(sub_sequence)
                sub_sequence = np.expand_dims(sub_sequence, axis=0)
                if self.verbose:
                    print("Sub seq shape: {0}".format(sub_sequence.shape))
                outputs.append(self.forward_duration(sub_sequence, alpha, non_empty_symbols=non_empty_symbols))

                if self.verbose:
                    print("Sub AEmb: {0}".format(outputs[-1].shape))

            aligned_emb = np.concatenate(outputs, axis=1)
        mels = []
        n_iters = aligned_emb.shape[1] // self.forward_len + 1
        for i in range(n_iters):
           start_idx = i * self.forward_len
           end_idx = min((i+1) * self.forward_len, aligned_emb.shape[1])
           if start_idx >= aligned_emb.shape[1]:
               break
           sub_aligned_emb = aligned_emb[:, start_idx:end_idx, :]
           if sub_aligned_emb.shape[1] < self.forward_len:
               sub_aligned_emb = np.pad(sub_aligned_emb, ((0, 0), (0, self.forward_len - sub_aligned_emb.shape[1]), (0, 0)), 'constant', constant_values=0)
           if self.verbose:
               print("SAEmb shape: {0}".format(sub_aligned_emb.shape))
           mel = self.forward_forward(sub_aligned_emb)[:, :end_idx - start_idx]
           mels.append(mel)

        res = np.concatenate(mels, axis=1)
        if self.verbose:
            print("MEL shape :{0}".format(res.shape))

        return res






