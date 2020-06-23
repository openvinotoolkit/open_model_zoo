from datetime import datetime
from argparse import ArgumentParser, ArgumentError, SUPPRESS
import logging
import sys
import wave

import numpy as np
from openvino.inference_engine import IECore


def type_overlap(val):
    if isinstance(val, str):
        if val.endswith('%'):
            try:
                res = float(val[:-1]) / 100
            except FloatingPointError:
                raise ArgumentError("Wrong value for '--overlap' argument")
        else:
            try:
                res = int(val)
            except ValueError:
                raise ArgumentError("Wrong value for '--overlap' argument")
    else:
        try:
            res = int(val)
        except ValueError:
            raise ArgumentError("Wrong value for '--overlap' argument")
    return res


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')

    args.add_argument('-i', '--input', type=str, required=True,
                      help="Required. Input to process")
    args.add_argument('-m', "--model", type=str, required=True,
                      help="Required. Path to an .xml file with a trained model.")
    args.add_argument("-l", "--cpu_extension", type=str, default=None,
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with "
                           "the kernels implementations.")
    args.add_argument("-d", "--device", type=str, default="CPU",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is"
                           " acceptable. The sample will look for a suitable plugin for device specified. "
                           "Default value is CPU")
    args.add_argument('--labels', type=str, default=None,
                      help="Optional. Labels mapping file")
    args.add_argument('-fr', '--framerate', type=int,
                      help="Optional. Set framerate for audio input")
    args.add_argument('-ol', '--overlap', type=type_overlap, default=0,
                      help='Optional. Set the overlapping between audio clip in samples or percent')

    return parser.parse_args()


class AudioSource:
    def __init__(self, source, channels=2, framerate=None):
        self.source = source
        self.framerate = framerate
        self.channels = channels

    def load(self):
        framerate, audio = read_wav(self.source, as_float=True)
        audio = audio.T
        if audio.shape[0] != self.channels:
            raise RuntimeError("Audio channels incorrect")
        if self.framerate:
            if self.framerate != framerate:
                audio = resample(audio, framerate, self.framerate)
        else:
            self.framerate = framerate

        self.audio = audio

    def chunks(self, size, hop=None, num_chunks=1):
        if not hop:
            hop = size
        pos = 0

        while pos + (num_chunks-1)*hop + size <= self.audio.shape[1]:
            yield np.array([self.audio[:, pos+n*hop: pos+n*hop+size] for n in range(num_chunks)])
            pos += hop*num_chunks

        # while pos + size <= self.audio.shape[1]:
        #     yield self.audio[:, pos: pos+size]
        #     pos += hop


def resample(audio, sample_rate, new_sample_rate):
    duration = audio.shape[1] / float(sample_rate)
    x_old = np.linspace(0, duration, audio.shape[1])
    x_new = np.linspace(0, duration, int(duration*new_sample_rate))
    data = np.array([np.interp(x_new, x_old, channel) for channel in audio])

    return data


def read_wav(file, as_float=False):
    sampwidth_types = {
        1: np.uint8,
        2: np.int16,
        3: np.int32,
        4: np.int32
    }
    sampwidth_max = {
        1: 255,
        2: 2**15,
        3: 2*23,
        4: 2**31
    }
    with wave.open(file, "rb") as wav:
        params = wav.getparams()
        data = wav.readframes(params.nframes)
        if sampwidth_types.get(params.sampwidth):
            data = np.frombuffer(data, dtype=sampwidth_types[params.sampwidth])
        else:
            raise RuntimeError("Couldn't process file {}: unsupported sample width {}"
                               .format(str(file), params.sampwidth))
        data = np.reshape(data, (params.nframes, params.nchannels))
        if as_float:
            data = data / sampwidth_max[params.sampwidth]

    return params.framerate, data


def main():
    args = build_argparser()

    logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
    log = logging.getLogger()

    log.info("Creating Inference Engine")
    ie = IECore()

    if args.device == "CPU" and args.cpu_extension:
        ie.add_extension(args.cpu_extension, 'CPU')

    log.info("Loading model {}".format(args.model))
    model_path = args.model[:-4]
    net = ie.read_network(model_path + ".xml", model_path + ".bin")

    if args.device == "CPU":
        supported_layers = ie.query_network(net, args.device)
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) > 0:
            raise RuntimeError("Following layers are not supported by the {} plugin:\n {}"
                               .format(args.device, ', '.join(not_supported_layers)))

    if len(net.inputs) != 1:
        log.error("Demo supports only models with 1 input layer")
        sys.exit(1)
    input_blob = next(iter(net.inputs))
    input_shape = net.inputs[input_blob].shape
    if len(net.outputs) != 1:
        log.error("Demo supports only models with 1 output layer")
        sys.exit(1)
    output_blob = next(iter(net.outputs))

    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)

    log.info("Preparing input")

    labels = []
    if args.labels:
        with open(args.labels, "r") as file:
            labels = [l.rstrip() for l in file.readlines()]

    batch_size, channels, _, length = input_shape

    audio = AudioSource(args.input, channels=channels, framerate=args.framerate)
    audio.load()

    hop = length - args.overlap if isinstance(args.overlap, int) else int(length * (1.0 - args.overlap))
    if hop < 0:
        log.error("Wrong value for '--overlap' argument - overlapping more than clip length")
        sys.exit(1)

    log.info("Starting inference")
    outputs = []
    clips = 0
    infer_time = []
    for id, chunk in enumerate(audio.chunks(length, hop, num_chunks=batch_size)):
        if len(chunk.shape) != len(input_shape):
            chunk = np.reshape(chunk, newshape=input_shape)
        infer_start_time = datetime.now()
        output = exec_net.infer(inputs={input_blob: chunk})
        infer_time.append(datetime.now() - infer_start_time)
        clips += batch_size
        output = output[output_blob]
        for batch, data in enumerate(output):
            start_time = (id*batch_size + batch)*hop / audio.framerate
            end_time = ((id*batch_size + batch)*hop + length) / audio.framerate
            outputs.append(data)
            label = np.argmax(data)
            log.info("[{:.2f}:{:.2f}] - {:s}: {:.2f}%".format(start_time, end_time,
                                                              labels[label] if labels else "Class {}".format(label),
                                                              data[label] * 100))

    if clips == 0:
        log.error("Audio too short for inference by that model")
        sys.exit(1)
    total = np.mean(outputs, axis=0)
    label = np.argmax(total)
    log.info("Total over audio - {:s}: {:.2f}%".format(labels[label] if labels else "Class {}".format(label),
                                                       total[label]*100))
    logging.info("Average infer time - {:.3f}s per clip".format((np.array(infer_time).sum() / clips).total_seconds()))


if __name__ == '__main__':
    main()
