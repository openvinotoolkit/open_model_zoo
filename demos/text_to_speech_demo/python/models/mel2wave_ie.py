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

import logging as log
import os.path as osp

import numpy as np

from utils.wav_processing import (
    fold_with_overlap, infer_from_discretized_mix_logistic, pad_tensor, xfade_and_unfold,
)


class WaveRNNIE:
    def __init__(self, model_upsample, model_rnn, ie, target=11000, overlap=550, hop_length=275, bits=9, device='CPU',
                 verbose=False, upsampler_width=-1):
        """
        return class provided WaveRNN inference.

        :param model_upsample: path to xml with upsample model of WaveRNN
        :param model_rnn: path to xml with rnn parameters of WaveRNN model
        :param target: length of the processed fragments
        :param overlap: overlap of the processed frames
        :param hop_length: The number of samples between successive frames, e.g., the columns of a spectrogram.
        :return:
        """
        self.verbose = verbose
        self.device = device
        self.target = target
        self.overlap = overlap
        self.dynamic_overlap = overlap
        self.hop_length = hop_length
        self.bits = bits
        self.indent = 550
        self.pad = 2
        self.batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        self.ie = ie

        self.upsample_net = self.load_network(model_upsample)
        if upsampler_width > 0:
            orig_shape = self.upsample_net.input_info['mels'].input_data.shape
            self.upsample_net.reshape({"mels": (orig_shape[0], upsampler_width, orig_shape[2])})

        self.upsample_exec = self.create_exec_network(self.upsample_net, model_upsample)

        self.rnn_net = self.load_network(model_rnn)
        self.rnn_exec = self.create_exec_network(self.rnn_net, model_rnn, batch_sizes=self.batch_sizes)

        # fixed number of the mels in mel-spectrogramm
        self.mel_len = self.upsample_net.input_info['mels'].input_data.shape[1] - 2 * self.pad
        self.rnn_width = self.rnn_net.input_info['x'].input_data.shape[1]

    def load_network(self, model_xml):
        model_bin_name = ".".join(osp.basename(model_xml).split('.')[:-1]) + ".bin"
        model_bin = osp.join(osp.dirname(model_xml), model_bin_name)
        log.info('Reading WaveRNN model {}'.format(model_xml))
        net = self.ie.read_network(model=model_xml, weights=model_bin)
        return net

    def create_exec_network(self, net, path, batch_sizes=None):
        if batch_sizes is not None:
            exec_net = []
            for b_s in batch_sizes:
                net.batch_size = b_s
                exec_net.append(self.ie.load_network(network=net, device_name=self.device))
        else:
            exec_net = self.ie.load_network(network=net, device_name=self.device)
        log.info('The WaveRNN model {} is loaded to {}'.format(path, self.device))
        return exec_net

    @staticmethod
    def get_rnn_init_states(b_size=1, rnn_dims=328):
        h1 = np.zeros((b_size, rnn_dims), dtype=float)
        h2 = np.zeros((b_size, rnn_dims), dtype=float)
        x = np.zeros((b_size, 1), dtype=float)
        return h1, h2, x

    def forward(self, mels):
        mels = (mels + 4) / 8
        np.clip(mels, 0, 1, out=mels)
        mels = np.transpose(mels)
        mels = np.expand_dims(mels, axis=0)

        n_parts = mels.shape[1] // self.mel_len + 1 if mels.shape[1] % self.mel_len > 0 else mels.shape[
                                                                                                 1] // self.mel_len
        upsampled_mels = []
        aux = []
        last_padding = 0
        for i in range(n_parts):
            i_start = i * self.mel_len
            i_end = i_start + self.mel_len
            if i_end > mels.shape[1]:
                last_padding = i_end - mels.shape[1]
                mel = np.pad(mels[:, i_start:mels.shape[1], :], ((0, 0), (0, last_padding), (0, 0)), 'constant',
                             constant_values=0)
            else:
                mel = mels[:, i_start:i_end, :]

            upsampled_mels_b, aux_b = self.forward_upsample(mel)
            upsampled_mels.append(upsampled_mels_b)
            aux.append(aux_b)
        if len(aux) > 1:
            upsampled_mels = np.concatenate(upsampled_mels, axis=1)
            aux = np.concatenate(aux, axis=1)
        else:
            upsampled_mels = upsampled_mels[0]
            aux = aux[0]
        if last_padding > 0:
            upsampled_mels = upsampled_mels[:, :-last_padding * self.hop_length, :]
            aux = aux[:, :-last_padding * self.hop_length, :]

        upsampled_mels, (_, self.dynamic_overlap) = fold_with_overlap(upsampled_mels, self.target, self.overlap)
        aux, _ = fold_with_overlap(aux, self.target, self.overlap)

        audio = self.forward_rnn(mels, upsampled_mels, aux)
        audio = (audio * (2 ** 15 - 1)).astype("<h")

        return audio

    def forward_upsample(self, mels):
        mels = pad_tensor(mels, pad=self.pad)

        out = self.upsample_exec.infer(inputs={"mels": mels})
        upsample_mels, aux = out["upsample_mels"][:, self.indent:-self.indent, :], out["aux"]
        return upsample_mels, aux

    def forward_rnn(self, mels, upsampled_mels, aux):
        wave_len = (mels.shape[1] - 1) * self.hop_length

        d = aux.shape[2] // 4
        aux_split = [aux[:, :, d * i:d * (i + 1)] for i in range(4)]

        b_size, seq_len, _ = upsampled_mels.shape
        seq_len = min(seq_len, aux_split[0].shape[1])

        if b_size not in self.batch_sizes:
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


class MelGANIE:
    def __init__(self, model, ie, device='CPU', default_width=800):
        """
        return class provided MelGAN inference.

        :param model: path to xml with MelGAN model of WaveRNN
        :param ie: instance of the IECore
        :param device: target device
        :return:
        """
        self.device = device
        self.ie = ie

        self.scales = 4
        self.hop_length = 256

        self.net = self.load_network(model)
        if self.net.input_info['mel'].input_data.shape[2] != default_width:
            orig_shape = self.net.input_info['mel'].input_data.shape
            new_shape = (orig_shape[0], orig_shape[1], default_width)
            self.net.reshape({"mel": new_shape})

        self.exec_net = self.create_exec_network(self.net, self.scales)

        # fixed number of columns in mel-spectrogramm
        self.mel_len = self.net.input_info['mel'].input_data.shape[2]
        self.widths = [self.mel_len * (i + 1) for i in range(self.scales)]

    def load_network(self, model_xml):
        model_bin_name = ".".join(osp.basename(model_xml).split('.')[:-1]) + ".bin"
        model_bin = osp.join(osp.dirname(model_xml), model_bin_name)
        log.info('Reading MelGAN model {}'.format(model_xml))
        net = self.ie.read_network(model=model_xml, weights=model_bin)
        return net

    def create_exec_network(self, net, path, scales=None):
        if scales is not None:
            orig_shape = net.input_info['mel'].input_data.shape
            exec_net = []
            for i in range(scales):
                new_shape = (orig_shape[0], orig_shape[1], orig_shape[2] * (i + 1))
                net.reshape({"mel": new_shape})
                exec_net.append(self.ie.load_network(network=net, device_name=self.device))
                net.reshape({"mel": orig_shape})
        else:
            exec_net = self.ie.load_network(network=net, device_name=self.device)
        log.info('The MelGAN model {} is loaded to {}'.format(path, self.device))
        return exec_net

    def forward(self, mel):
        mel = np.expand_dims(mel, axis=0)
        res_audio = []
        last_padding = 0
        if mel.shape[2] % self.mel_len:
            last_padding = self.mel_len - mel.shape[2] % self.mel_len

        mel = np.pad(mel, ((0, 0), (0, 0), (0, last_padding)), 'constant', constant_values=-11.5129)

        active_net = -1
        cur_w = -1
        cols = mel.shape[2]

        for i, w in enumerate(self.widths):
            if cols <= w:
                cur_w = w
                active_net = i
                break
        if active_net == -1:
            cur_w = self.widths[-1]

        c_begin = 0
        c_end = cur_w
        while c_begin < cols:
            audio = self.exec_net[active_net].infer(inputs={"mel": mel[:, :, c_begin:c_end]})["audio"]
            res_audio.extend(audio)

            c_begin = c_end

            if c_end + cur_w >= cols:
                for i, w in enumerate(self.widths):
                    if w >= cols - c_end:
                        cur_w = w
                        active_net = i
                        break

            c_end += cur_w
        if last_padding:
            audio = res_audio[:-self.hop_length * last_padding]
        else:
            audio = res_audio

        audio = np.array(audio).astype(dtype=np.int16)

        return audio
