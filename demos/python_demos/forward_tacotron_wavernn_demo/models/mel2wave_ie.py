import sys
from openvino.inference_engine import IECore
import numpy as np
import logging as log
import time
import os.path as osp
import pickle

from utils.wav_processing import *
import openvino


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('{0}  {1} ms'.format(method.__name__, (te - ts) * 1000))
        return result
    return timed


class Mel2WaveInference():
    def __init__(self):
        self.ie = IECore()

    def load_network(self, model_xml, batch_sizes=None):
        model_bin_name = ".".join(osp.basename(model_xml).split('.')[:-1]) + ".bin"
        model_bin = osp.join(osp.dirname(model_xml), model_bin_name)
        # Read IR
        log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
        net = self.ie.read_network(model=model_xml, weights=model_bin)


        print("#################################################################")
        print("Model: {0}. Inputs: {1}".format(model_xml, net.inputs))
        print("#################################################################")
        print("Model: {0}. Outputs: {1}".format(model_xml, net.outputs))
        if batch_sizes is not None:
            exec_net = []
            for b_s in batch_sizes:
                net.batch_size = b_s
                exec_net.append(self.ie.load_network(network=net, device_name=self.device))
        else:
            exec_net = self.ie.load_network(network=net, device_name=self.device)
        return net, exec_net


class WaveRNNIE(Mel2WaveInference):
    def __init__(self, model_upsample, model_rnn, target=11000, overlap=550, hop_length=275, bits=9, device='CPU', verbose=False):
        """
        return class provided WaveRNN inference.

        :param model_upsample: path to xml with upsample model of WaveRNN
        :param model_rnn: path to xml with rnn parameters of WaveRNN model
        :param target: length of the processed fragments
        :param overlap: overlap of the processed frames
        :param hop_length: The number of samples between successive frames, e.g., the columns of a spectrogram.
        :return:
        """
        super().__init__()
        self.verbose = verbose
        self.device = device
        self.target = target
        self.overlap = overlap
        self.dynamic_overlap = overlap
        self.hop_length = hop_length
        self.bits = bits
        self.indent = 550
        self.pad = 2
        self.batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256]

        self.ie = IECore()

        self.upsample_net, self.upsample_exec = self.load_network(model_upsample)
        self.rnn_net, self.rnn_exec = self.load_network(model_rnn, batch_sizes=self.batch_sizes)

        # fixed number of the mels in mel-spectrogramm
        self.mel_len = self.upsample_net.inputs['mels'].shape[1] - 2 * self.pad
        self.rnn_width = self.rnn_net.inputs['x'].shape[1]

    def get_rnn_init_states(self, b_size=1, rnn_dims=328):
        h1 = np.zeros((b_size, rnn_dims), dtype=float)
        h2 = np.zeros((b_size, rnn_dims), dtype=float)
        x = np.zeros((b_size, 1), dtype=float)
        return h1, h2, x

    def forward_fixed_shape(self, mels):
        upsampled_mels, aux = self.forward_upsample(mels)
        upsampled_mels = fold_with_overlap(upsampled_mels, self.target, self.overlap)
        aux = fold_with_overlap(aux, self.target, self.overlap)
        audio = self.forward_rnn(mels, upsampled_mels, aux)

        return audio

    def forward(self, mels):
        n_parts = mels.shape[1] // self.mel_len + 1 if mels.shape[1] % self.mel_len >0 else mels.shape[1] // self.mel_len
        upsampled_mels = []
        aux = []
        last_padding = 0
        for i in range(n_parts):
            i_start = i * self.mel_len
            i_end = i_start + self.mel_len
            if i_end > mels.shape[1]:
                last_padding = i_end - mels.shape[1]
                mel = np.pad(mels[:, i_start:mels.shape[1], :], ((0, 0), (0, last_padding), (0, 0)), 'constant', constant_values=0)
            else:
                mel = mels[:,i_start:i_end, :]

            upsampled_mels_b, aux_b = self.forward_upsample(mel)
            upsampled_mels.append(upsampled_mels_b)
            aux.append(aux_b)
        if len(aux) > 1:
            upsampled_mels = np.concatenate(upsampled_mels,axis=1)
            aux = np.concatenate(aux, axis=1)
        else:
            upsampled_mels = upsampled_mels[0]
            aux = aux[0]
        if last_padding > 0:
            upsampled_mels = upsampled_mels[:,:-last_padding*self.hop_length,:]
            aux = aux[:,:-last_padding*self.hop_length,:]

        upsampled_mels, (_, self.dynamic_overlap) = fold_with_overlap(upsampled_mels, self.target, self.overlap)
        aux, _ = fold_with_overlap(aux, self.target, self.overlap)

        audio = self.forward_rnn(mels, upsampled_mels, aux)

        return audio

    @timeit
    def forward_upsample(self, mels):
        mels = pad_tensor(mels, pad=self.pad)

        out = self.upsample_exec.infer(inputs={"mels": mels})
        upsample_mels, aux = out["upsample_mels"][:,self.indent:-self.indent,:], out["aux"]
        return upsample_mels, aux

    def forward_upsample_one_iter(self, inputs):
        out = self.upsample_exec.infer(inputs=inputs)
        return out
    @timeit
    def forward_rnn(self, mels, upsampled_mels, aux):
        wave_len = (mels.shape[1] - 1) * self.hop_length

        d = aux.shape[2] // 4
        aux_split = [aux[:, :, d * i:d * (i + 1)] for i in range(4)]

        b_size, seq_len, _ = upsampled_mels.shape

        if not b_size in self.batch_sizes:
            raise Exception('Incorrect batch size {0}. Correct should be 2 ** something'.format(b_size))

        active_network = self.batch_sizes.index(b_size)


        h1, h2, x = self.get_rnn_init_states(b_size, self.rnn_width)

        output = []

        for i in range(seq_len):
            m_t = upsampled_mels[:, i, :]

            a1_t, a2_t, a3_t, a4_t = \
                (a[:, i, :] for a in aux_split)

            out = self.rnn_exec[active_network].infer(inputs={"m_t": m_t, "a1_t": a1_t, "a2_t": a2_t, "a3_t": a3_t,
                                              "a4_t": a4_t, "h1.1": h1, "h2.1": h2, "x": x})

            logits = out["logits"]
            h1 = out["h1"]
            h2 = out["h2"]

            sample = infer_from_discretized_mix_logistic(logits)

            x = sample[:]
            x = np.expand_dims(x, axis=1)
            output.append(sample)

        output = np.stack(output).transpose(1, 0)
        output = output.astype(np.float64)

        if b_size > 1:
            output = xfade_and_unfold(output, self.dynamic_overlap)
        else:
            output = output[0]

        fade_out = np.linspace(1, 0, 20 * self.hop_length)
        output = output[:wave_len]
        output[-20 * self.hop_length:] *= fade_out
        return output

    def forward_rnn_one_iter(self, inputs):
        out = self.rnn_exec.infer(inputs=inputs)
        return out


    def perf_count(self, mels, rnn_inputs):
        req_mels = mels
        self.upsample_exec.requests[0].infer(inputs=req_mels)
        perf_res = self.upsample_exec.requests[0].get_perf_counts()

        layer_type_times = {}
        print("########## Per layer stats UPSAMPLE ##############")
        for key, val in perf_res.items():
            print(key, val)
            l_type = val['layer_type']
            l_time = val['cpu_time']
            if l_type in layer_type_times:
                layer_type_times[l_type] += l_time
            else:
                layer_type_times[l_type] = l_time
        print("########## Per type stats UPSAMPLE ##############")
        for key, val in layer_type_times.items():
            print(key, val)

        self.rnn_exec.requests[0].infer(inputs=rnn_inputs)
        perf_res = self.rnn_exec.requests[0].get_perf_counts()

        layer_type_times = {}
        print("########## Per layer stats RNN ##############")
        for key, val in perf_res.items():
            print(key, val)
            l_type = val['layer_type']
            l_time = val['cpu_time']
            if l_type in layer_type_times:
                layer_type_times[l_type] += l_time
            else:
                layer_type_times[l_type] = l_time
        print("########## Per type stats RNN ##############")
        for key, val in layer_type_times.items():
            print(key, val)
