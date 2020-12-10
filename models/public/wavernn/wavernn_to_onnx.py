#!/usr/bin/env python3
import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from models.fatchord_version import WaveRNN
from utils import hparams as hp

##################################################################################################

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
        mels = mels.transpose(1, 2)
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
        implement one step forward pass from WaveRNN fatchord version
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

##################################################################################################

def main():
    # Parse Arguments
    parser = argparse.ArgumentParser(description='TTS Generator')

    parser.add_argument('--mel', type=str, help='[string/path] path to test mel file')

    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py',
                        help='The file to use for the hyperparameters')

    parser.add_argument('--batched', '-b', dest='batched', action='store_true', help='Fast Batched Generation')

    parser.add_argument('--voc_weights', type=str, help='[string/path] Load in different FastSpeech weights', default="pretrained/wave_800K.pyt")

    args = parser.parse_args()

    if not os.path.exists('onnx'):
        os.mkdir('onnx')

    hp.configure(args.hp_file)

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


    opset_version = 11

    with torch.no_grad():
        mels = np.load(args.mel)
        mels = torch.from_numpy(mels)
        mels = mels.unsqueeze(0)
        mels = voc_upsampler.pad_tensor(mels)

        mels_onnx = mels.clone()

        torch.onnx.export(voc_upsampler, mels_onnx, "./onnx/wavernn_upsampler.onnx",
                          opset_version=opset_version,
                          do_constant_folding=True,
                          input_names=["mels"],
                          output_names=["upsample_mels", "aux"])


        mels, aux = voc_upsampler(mels)
        mels = mels[:, 550:-550, :]

        mels, aux = voc_upsampler.fold(mels, aux)

        h1, h2, x = voc_infer.get_initial_parameters(mels)

        aux_split = voc_infer.split_aux(aux)

        b_size, seq_len, _ = mels.size()

        if seq_len:
            m_t = mels[:, 0, :]

            a1_t, a2_t, a3_t, a4_t = \
                (a[:, 0, :] for a in aux_split)

            rnn_input = (m_t, a1_t, a2_t, a3_t, a4_t, h1, h2, x)
            torch.onnx.export(voc_infer, rnn_input, "./onnx/wavernn_rnn.onnx",
                              opset_version=opset_version,
                              do_constant_folding=True,
                              input_names=["m_t", "a1_t", "a2_t", "a3_t", "a4_t", "h1", "h2", "x"],
                              output_names=["logits", "h1", "h2"])

    print('Done!')

if __name__ == '__main__':
    main()
