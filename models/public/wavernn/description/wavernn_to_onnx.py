import os
import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.distribution import sample_from_discretized_mix_logistic
from models.fatchord_version_fused import WaveRNN

from utils import hparams as hp
from utils.paths import Paths
import argparse

from utils.dsp import reconstruct_waveform, save_wav
from utils.dsp import *

import numpy as np
import pickle

##################################################################################################

def forwardLikeGRUCell(input, h, gru):
    r"""A gated recurrent unit (GRU) cell

        .. math::

            \begin{array}{ll}
            r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
            z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
            n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
            h' = (1 - z) * n + z * h
            \end{array}
        """

    #w_ir, w_ii, w_ih = gru.weight_ih_l0.chunk(0, 3)
    #w_ir, w_ii, w_ih = gru.weight_hh_l0.chunk(0, 3)
    w_ih = gru.weight_ih_l0
    w_hh = gru.weight_hh_l0
    b_ih = gru.bias_ih_l0
    b_hh = gru.bias_hh_l0

    i_s = F.linear(input, w_ih, b_ih)
    h_s = F.linear(h, w_hh, b_hh)

    r_i_s, z_i_s, o_i_s = i_s.chunk(3, 1)
    r_h_s, z_h_s, o_h_s = h_s.chunk(3, 1)

    r = torch.sigmoid(r_i_s + r_h_s)
    z = torch.sigmoid(z_i_s + z_h_s)
    n = torch.tanh(o_i_s + r * o_h_s)

    h_t = (1 - z) * n + z * h

    return h_t

class WaveRNNUpsamplerONNX(nn.Module):
    def __init__(self, model, batched, target, overlap):
        super().__init__()
        model.upsample.fuse()
        self.model = model
        self.batched = batched
        self.target = target
        self.overlap = overlap

    def pad_tensor(self, mels):
        mels = self.model.pad_tensor(mels.transpose(1, 2), pad=self.model.pad, side='both')
        return mels

    def fold(self, mels, aux):
        if self.batched:
            mels = self.model.fold_with_overlap(mels, self.target, self.overlap)
            aux = self.model.fold_with_overlap(aux, self.target, self.overlap)
        return mels, aux

    def forward(self, mels):
        mels = mels.transpose(1,2)
        aux = self.model.upsample.resnet(mels)
        aux = aux.unsqueeze(1)
        aux = self.model.upsample.resnet_stretch(aux)
        aux = aux.squeeze(1)
        upsample_mels = mels.unsqueeze(1)
        for f in self.model.upsample.up_layers:
            upsample_mels = f(upsample_mels)

        return upsample_mels.squeeze(1).transpose(1, 2), aux.transpose(1, 2)


class WaveRNNONNX(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.rnn1 = self.model.get_gru_cell(self.model.rnn1)
        self.rnn2 = self.model.get_gru_cell(self.model.rnn2)

    def get_initial_parameters(self, mels):
        device = next(self.model.parameters()).device
        b_size, seq_len, _ = mels.size()

        h1 = torch.zeros(b_size, self.model.rnn_dims, device=device)
        h2 = torch.zeros(b_size, self.model.rnn_dims, device=device)
        x = torch.zeros(b_size, 1, device=device)

        return h1, h2, x

    def split_aux(self, aux):
        d = self.model.aux_dims
        aux_split = [aux[:, :, d * i:d * (i + 1)] for i in range(4)]
        return aux_split

    def forward(self, m_t, a1_t, a2_t, a3_t, a4_t, h1, h2, x):
        return self.infer(m_t, a1_t, a2_t, a3_t, a4_t, h1, h2, x)

    def infer(self, m_t, a1_t, a2_t, a3_t, a4_t, h1, h2, x):
        """
        implement one step forward pass form WaveRNN fachord version
        :return:
        """
        x = torch.cat([x, m_t, a1_t], dim=1)
        x = self.model.I(x)
        h1 = self.rnn1(x, h1)

        x = x + h1
        inp = torch.cat([x, a2_t], dim=1)
        h2 = self.rnn2(inp, h2)

        x = x + h2
        x = torch.cat([x, a3_t], dim=1)
        x = F.relu(self.model.fc1(x))

        x = torch.cat([x, a4_t], dim=1)
        x = F.relu(self.model.fc2(x))

        logits = self.model.fc3(x)

        return logits, h1, h2



    def infer_from_logits(self, logits):
        if self.model.mode == 'MOL':
            sample = sample_from_discretized_mix_logistic(logits.unsqueeze(0).transpose(1, 2))
        elif self.model.mode == 'RAW':
            posterior = F.softmax(logits, dim=1)
            distrib = torch.distributions.Categorical(posterior)

            sample = 2 * distrib.sample().float() / (self.model.n_classes - 1.) - 1.
        else:
            raise RuntimeError("Unknown model mode value - ", self.mode)
        return sample

    def xfade_and_unfold(self, y, target, overlap):
        if self.model.mode == 'RAW':
            y = decode_mu_law(y, self.model.n_classes, False)
        return self.model.xfade_and_unfold(y, target, overlap)

##################################################################################################
class DurationPredictor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, alpha=1.0):
        x = self.model.embedding(x)
        dur = self.model.dur_pred(x, alpha=alpha)

        x = x.transpose(1, 2)
        x = self.model.prenet(x)

        return x, dur


    def lr(self, x, dur):
        return self.model.lr(x, dur)




class Tacotorn(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x, _ = self.model.lstm(x)

        x = self.model.lin(x)
        x = x.transpose(1, 2)

        x_post = self.model.postnet(x)
        x_post = self.model.post_proj(x_post)
        x_post = x_post.transpose(1, 2)

        x_post = x_post.squeeze()

        return x_post


def main():
    # Parse Arguments
    parser = argparse.ArgumentParser(description='TTS Generator')

    parser.add_argument('--mel', type=str, help='[string/path] path to test mel file')

    parser.add_argument('--force_cpu', '-c', action='store_true',
                        help='Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py',
                        help='The file to use for the hyperparameters')

    parser.add_argument('--batched', '-b', dest='batched', action='store_true', help='Fast Batched Generation')

    parser.add_argument('--voc_weights', type=str, help='[string/path] Load in different FastSpeech weights', default="pretrained/wave_800K.pyt")

    parser.add_argument('--voc_onnx', dest='voc_onnx', action='store_true', help='Convert or not vocoder ot onnx')

    args = parser.parse_args()

    if not os.path.exists('onnx'):
        os.mkdir('onnx')

    hp.configure(args.hp_file)

    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    #####
    print('\nInitialising WaveRNN Model...\n')
    # Instantiate WaveRNN Model
    voc_model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                        fc_dims=hp.voc_fc_dims,
                        bits=hp.bits,
                        pad=hp.voc_pad,
                        upsample_factors=hp.voc_upsample_factors,
                        feat_dims=hp.num_mels,
                        compute_dims=hp.voc_compute_dims,
                        res_out_dims=hp.voc_res_out_dims,
                        res_blocks=hp.voc_res_blocks,
                        hop_length=hp.hop_length,
                        sample_rate=hp.sample_rate,
                        mode=hp.voc_mode).to(device)

    voc_load_path = args.voc_weights
    voc_model.load(voc_load_path)

    voc_upsampler = WaveRNNUpsamplerONNX(voc_model, args.batched, hp.voc_target, hp.voc_overlap)
    voc_infer = WaveRNNONNX(voc_model)

    voc_model.eval()
    voc_upsampler.eval()
    voc_infer.eval()


    opset_version = 10

    with torch.no_grad():
        mels = np.load(args.mel)

        mel = (mels + 4) / 8
        np.clip(mel, 0, 1, out=mel)
        wav = reconstruct_waveform(mels, n_iter=60)

        save_wav(wav, "./onnx/wav_from_mel_griffin_lim.wav")

        '''
        FIRST STEP: Upsample mels to the output wave shape
        '''
        mels = torch.from_numpy(mels)
        mels = mels.unsqueeze(0)

        wave_len = (mels.size(-1) - 1) * voc_model.hop_length

        mels = voc_upsampler.pad_tensor(mels)

        mels_onnx = mels.clone()

        print(mels_onnx.shape)
        torch.onnx.export(voc_upsampler, mels_onnx, "./onnx/wavernn_upsampler.onnx",
                          opset_version=opset_version,
                          do_constant_folding=True,
                          input_names=["mels"],
                          output_names=["upsample_mels", "aux"])


        mels, aux = voc_upsampler(mels)
        mels = mels[:, 550:-550, :]

        mels, aux = voc_upsampler.fold(mels, aux)

        print("Mels/Aux fold shape: {0}/{1}".format(mels.shape, aux.shape))

        output = []

        h1, h2, x = voc_infer.get_initial_parameters(mels)

        aux_split = voc_infer.split_aux(aux)

        b_size, seq_len, _ = mels.size()

        '''
        SECOND STEP: Iterativelly generate wavs
        '''

        for i in range(seq_len):
            m_t = mels[:, i, :]

            a1_t, a2_t, a3_t, a4_t = \
                (a[:, i, :] for a in aux_split)

            if i == 0:
                rnn_input = (m_t, a1_t, a2_t, a3_t, a4_t, h1, h2, x)
                torch.onnx.export(voc_infer, rnn_input, "./onnx/wavernn_rnn.onnx",
                                  opset_version=opset_version,
                                  do_constant_folding=True,
                                  input_names=["m_t", "a1_t", "a2_t", "a3_t", "a4_t", "h1", "h2", "x"],
                                  output_names=["logits", "h1", "h2"])


            logits, h1, h2 = voc_infer(m_t, a1_t, a2_t, a3_t, a4_t, h1, h2, x)

            sample = voc_infer.infer_from_logits(logits)

            x = sample.transpose(0, 1)

            output.append(sample.view(-1))

        output = torch.stack(output).transpose(0, 1)
        output = output.cpu().numpy()
        output = output.astype(np.float64)

        if args.batched:
            output = voc_infer.xfade_and_unfold(output, hp.voc_target, hp.voc_overlap)
        else:
            output = output[0]

        # Fade-out at the end to avoid signal cutting out suddenly
        fade_out = np.linspace(1, 0, 20 * voc_model.hop_length)
        output = output[:wave_len]
        output[-20 * voc_model.hop_length:] *= fade_out

        save_wav(output, 'onnx/wav_from_mel_wavernn.wav')

    print('Done!')

if __name__ == '__main__':
    main()
