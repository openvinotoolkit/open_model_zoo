import os

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.forward_tacotron import ForwardTacotron
from utils.distribution import sample_from_discretized_mix_logistic

from utils import hparams as hp
from utils.text.symbols import symbols
from utils.paths import Paths
import argparse
from utils.text import text_to_sequence
import numpy as np

##################################################################################################

class CBHGWrapper(nn.Module):
    """Class for changing forward pass in CBHG layer: MaxPool1d to MaxPool2d."""
    def __init__(self, model):
        super().__init__()

        # List of all rnns to call `flatten_parameters()` on
        self.cbhg = model
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 1), stride=1, padding=(1, 0))

    def forward(self, x):
        # Save these for later
        residual = x
        seq_len = x.size(-1)
        conv_bank = []

        # Convolution Bank
        for conv in self.cbhg.conv1d_bank:
            c = conv(x) # Convolution
            conv_bank.append(c[:, :, :seq_len])

        # Stack along the channel axis
        conv_bank = torch.cat(conv_bank, dim=1)

        # dump the last padding to fit residual
        conv_bank = conv_bank.unsqueeze(-1)
        x = self.maxpool(conv_bank)
        x = x.squeeze(-1)
        x = x[:, :, :seq_len]

        # Conv1d projections
        x = self.cbhg.conv_project1(x)
        x = self.cbhg.conv_project2(x)

        # Residual Connect
        x = x + residual

        # Through the highways
        x = x.transpose(1, 2)
        if self.cbhg.highway_mismatch is True:
            x = self.cbhg.pre_highway(x)
        for h in self.cbhg.highways: x = h(x)

        x, _ = self.cbhg.rnn(x)
        return x


class DurationPredictor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.prenet = CBHGWrapper(self.model.prenet)

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
        self.model.postnet = CBHGWrapper(self.model.postnet)

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

    parser.add_argument('--tts_weights', type=str, help='[string/path] Load in different FastSpeech weights')

    parser.add_argument('--force_cpu', '-c', action='store_true',
                        help='Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py',
                        help='The file to use for the hyperparameters')
    parser.add_argument('--alpha', type=float, default=1.,
                        help='Parameter for controlling length regulator for speedup '
                             'or slow-down of generated speech, e.g. alpha=2.0 is double-time')

    if not os.path.exists('onnx'):
        os.mkdir('onnx')

    args = parser.parse_args()


    hp.configure(args.hp_file)

    input_text = "the forms of printed letters should be beautiful, and that their arrangement on the page should be reasonable and a help to the shapeliness of the letters themselves."
    tts_weights = args.tts_weights

    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    print('\nInitialising Forward TTS Model...\n')
    tts_model = ForwardTacotron(embed_dims=hp.forward_embed_dims,
                                num_chars=len(symbols),
                                durpred_rnn_dims=hp.forward_durpred_rnn_dims,
                                durpred_conv_dims=hp.forward_durpred_conv_dims,
                                rnn_dim=hp.forward_rnn_dims,
                                postnet_k=hp.forward_postnet_K,
                                postnet_dims=hp.forward_postnet_dims,
                                prenet_k=hp.forward_prenet_K,
                                prenet_dims=hp.forward_prenet_dims,
                                highways=hp.forward_num_highways,
                                dropout=hp.forward_dropout,
                                n_mels=hp.num_mels).to(device)

    tts_load_path = tts_weights if tts_weights else paths.forward_latest_weights
    tts_model.load(tts_load_path)


    encoder = DurationPredictor(tts_model)
    decoder = Tacotorn(tts_model)

    tts_model.eval()
    encoder.eval()
    decoder.eval()

    opset_version = 10

    with torch.no_grad():
        input_seq = text_to_sequence(input_text.strip(), hp.tts_cleaner_names)
        input_seq = torch.as_tensor(input_seq, dtype=torch.long, device=device).unsqueeze(0)

        '''
        FIRST STEP: predict symbols duration
        '''
        torch.onnx.export(encoder, input_seq,  "./onnx/forward_tacotron_duration_prediction.onnx",
                          opset_version=opset_version,
                          do_constant_folding=True,
                          input_names=["input_seq"],
                          output_names=["embeddings", "duration"])


        x, durations = encoder(input_seq)

        '''
        SECOND STEP: expand symbols by durations
        '''
        x = encoder.lr(x, durations)

        '''
        THIRD STEP: generate mel
        '''
        torch.onnx.export(decoder, x, "./onnx/forward_tacotron_regression.onnx",
                          opset_version=opset_version,
                          do_constant_folding=True,
                          input_names=["data"],
                          output_names=["mel"])

        mels = decoder(x)
        mel = mels.cpu().data.numpy()

        mel = (mel + 4) / 8
        np.clip(mel, 0, 1, out=mel)
        # Save if you want

    print('Done!')

if __name__ == '__main__':
    main()

