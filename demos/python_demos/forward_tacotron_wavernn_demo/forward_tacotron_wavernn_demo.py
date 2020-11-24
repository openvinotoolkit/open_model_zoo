#!/usr/bin/env python3

import sys
import time
from argparse import ArgumentParser, SUPPRESS

from tqdm import tqdm
import numpy as np
import scipy.io.wavfile
from openvino.inference_engine import IECore

from models.forward_tacotron_ie import ForwardTacotronIE
from models.mel2wave_ie import WaveRNNIE


def save_wav(x, path):
    sr = 22050
    scipy.io.wavfile.write(path, sr, x)


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m_duration", "--model_duration", help="Required. Path to ForwardTacotron`s duration prediction part (*.xml format).", required=True,
                      type=str)
    args.add_argument("-m_forward", "--model_forward", help="Required. Path to ForwardTacotron`s mel-spectrogram regression part (*.xml format).", required=True,
                      type=str)
    args.add_argument("-m_upsample", "--model_upsample", help="Required. Path to WaveRNN`s part for mel-spectrogram upsampling by time axis (*.xml format).", required=True,
                      type=str)
    args.add_argument("-m_rnn", "--model_rnn", help="Required. Path to WaveRNN`s part for waveform autoregression (*.xml format).", required=True,
                      type=str)

    args.add_argument("-i", "--input", help="Text file with text.", required=True,
                      type=str)
    args.add_argument("-o", "--out", help="Required. Path to an output .wav file", default='out.wav',
                      type=str)

    args.add_argument("--upsampler_width", default=-1,
                      help="Width for reshaping of the model_upsample. If -1 then no reshape. Do not use with FP16 model.",
                      required=False,
                      type=int)

    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
                           "acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU",
                      default="CPU", type=str)
    return parser


def mel_to_wave(mel_spec, out_name, ap):
    waveform = ap.reconstruct_waveform(mel_spec)
    ap.save_wav(waveform, out_name)


def main():
    args = build_argparser().parse_args()

    ie = IECore()
    vocoder = WaveRNNIE(args.model_upsample, args.model_rnn, ie, device=args.device, upsampler_width=args.upsampler_width)
    forward_tacotron = ForwardTacotronIE(args.model_duration, args.model_forward, ie, args.device, verbose=False)

    audio_res = []
    silent = np.array([1.0])

    len_th = 512

    time_forward = 0
    time_wavernn = 0

    time_s_all = time.perf_counter()
    with open(args.input, 'r') as f:
        count = 0
        for line in f:
            count += 1
            line = line.rstrip()
            print("Process line {0} with length {1}.".format(count, len(line)))

            if len(line) > len_th:
                texts = []
                prev_begin = 0
                delimers ='.!?;:'
                for i, c in enumerate(line):
                    if (c in delimers and i - prev_begin > len_th) or i == len(line) - 1:
                        texts.append(line[prev_begin:i+1])
                        prev_begin = i
            else:
                texts = [line]

            for text in tqdm(texts):
                time_s = time.perf_counter()
                mel = forward_tacotron.forward(text)
                time_e = time.perf_counter()
                time_forward += (time_e - time_s) * 1000

                mel = (mel + 4) / 8
                np.clip(mel, 0, 1, out=mel)

                if len(mel.shape) == 2:
                    mel = np.transpose(mel)

                    mel = np.expand_dims(mel, axis=0)

                time_s = time.perf_counter()
                audio = vocoder.forward(mel)
                time_e = time.perf_counter()
                time_wavernn += (time_e - time_s) * 1000

                audio_res.extend(audio)
                audio_res.extend(silent * min(audio))

            if count % 5 == 0:
                print('WaveRNN time: {:.3f}ms. ForwardTacotronTime {:.3f}ms'.format(time_wavernn, time_forward))
    time_e_all = time.perf_counter()

    print('All time {:.3f}ms. WaveRNN time: {:.3f}ms. ForwardTacotronTime {:.3f}ms'.format((time_e_all - time_s_all) * 1000,
                                                                                 time_wavernn, time_forward))

    save_wav(np.array(audio_res), args.out)


if __name__ == '__main__':
    sys.exit(main() or 0)
