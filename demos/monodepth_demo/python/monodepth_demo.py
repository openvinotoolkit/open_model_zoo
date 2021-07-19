#!/usr/bin/env python3

import sys
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IECore, get_version
import matplotlib.pyplot as plt

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


def main():
    # arguments
    parser = ArgumentParser()

    parser.add_argument(
        "-m", "--model", help="Required. Path to an .xml file with a trained model", required=True, type=Path)
    parser.add_argument(
        "-i", "--input", help="Required. Path to a input image file", required=True, type=str)
    parser.add_argument("-l", "--cpu_extension",
        help="Optional. Required for CPU custom layers. Absolute MKLDNN (CPU)-targeted custom layers. "
        "Absolute path to a shared library with the kernels implementations", type=str, default=None)
    parser.add_argument("-d", "--device",
        help="Optional. Specify the target device to infer on; CPU, GPU, HDDL or MYRIAD is acceptable. "
        "The demo will look for a suitable plugin for device specified. Default value is CPU", default="CPU", type=str)

    args = parser.parse_args()

    log.info('OpenVINO Inference Engine')
    log.info('\tbuild: {}'.format(get_version()))
    ie = IECore()
    if args.cpu_extension and "CPU" in args.device:
        ie.add_extension(args.cpu_extension, "CPU")

    log.info('Reading model {}'.format(args.model))
    net = ie.read_network(args.model, args.model.with_suffix(".bin"))

    assert len(net.input_info) == 1, "Expected model with only 1 input blob"
    assert len(net.outputs) == 1, "Expected model with only 1 output blob"

    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    net.batch_size = 1

    # read and pre-process input image
    _, _, height, width = net.input_info[input_blob].input_data.shape

    start_time = perf_counter()
    image = cv2.imread(args.input, cv2.IMREAD_COLOR)
    (input_height, input_width) = image.shape[:-1]

    # resize
    if (input_height, input_width) != (height, width):
        log.debug("Image is resized from {} to {}".format(
            image.shape[:-1], (height, width)))
        image = cv2.resize(image, (width, height), cv2.INTER_CUBIC)

    # prepare input
    image = image.astype(np.float32)
    image = image.transpose((2, 0, 1))
    image_input = np.expand_dims(image, 0)

    # loading model to the plugin
    exec_net = ie.load_network(network=net, device_name=args.device)
    log.info('The model {} is loaded to {}'.format(args.model, args.device))

    # start sync inference
    res = exec_net.infer(inputs={input_blob: image_input})

    # processing output blob
    disp = np.squeeze(res[out_blob][0])

    # resize disp to input resolution
    disp = cv2.resize(disp, (input_width, input_height), cv2.INTER_CUBIC)

    # rescale disp
    disp_min = disp.min()
    disp_max = disp.max()

    if disp_max - disp_min > 1e-6:
        disp = (disp - disp_min) / (disp_max - disp_min)
    else:
        disp.fill(0.5)

    total_latency = (perf_counter() - start_time) * 1e3
    log.info("Metrics report:")
    log.info("\tLatency: {:.1f} ms".format(total_latency))
    # pfm
    out = 'disp.pfm'
    cv2.imwrite(out, disp)

    log.debug("Disparity map was saved to {}".format(out))

    # png
    out = 'disp.png'
    plt.imsave(out, disp, vmin=0, vmax=1, cmap='inferno')

    log.debug("Color-coded disparity image was saved to {}".format(out))


if __name__ == '__main__':
    main()
