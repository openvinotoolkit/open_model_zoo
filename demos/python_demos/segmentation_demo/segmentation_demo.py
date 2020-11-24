#!/usr/bin/env python3
"""
 Copyright (C) 2018-2020 Intel Corporation

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

import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IECore

classes_color_map = [
    (150, 150, 150),
    (58, 55, 169),
    (211, 51, 17),
    (157, 80, 44),
    (23, 95, 189),
    (210, 133, 34),
    (76, 226, 202),
    (101, 138, 127),
    (223, 91, 182),
    (80, 128, 113),
    (235, 155, 55),
    (44, 151, 243),
    (159, 80, 170),
    (239, 208, 44),
    (128, 50, 51),
    (82, 141, 193),
    (9, 107, 10),
    (223, 90, 142),
    (50, 248, 83),
    (178, 101, 130),
    (71, 30, 204)
]

np.random.seed(10)


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("-i", "--input", help="Required. Path to a folder with images or path to an image files.",
                      required=True, type=str, nargs="+")
    args.add_argument("-lab", "--labels", help="Optional. Path to a text file containing class labels.",
                      type=str)
    args.add_argument("-c", "--colors", help="Optional. Path to a text file containing colors for classes.",
                      type=str)
    args.add_argument("-lw", "--legend_width", help="Optional. Width of legend.", default=300, type=int)
    args.add_argument("-o", "--output_dir", help="Optional. Path to a folder where output files will be saved.",
                      default="results", type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. "
                           "Absolute MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the "
                           "kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. Sample will look for a suitable plugin for device specified. Default value is CPU.",
                      default="CPU", type=str)
    return parser


def get_files(input_paths):
    inputs = []
    for path in input_paths:
        if not os.path.exists(path):
            raise AttributeError("Path to input data: '{}' does not exist".format(path))
        if os.path.isdir(path):
            file_paths = [os.path.join(path, file) for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
            inputs.extend(file_paths)
        else:
            inputs.append(path)
    return inputs


def get_info_from_file(file):
    if file is None:
        return None
    info = []
    with open(file, 'r') as file:
        for line in file.readlines():
            info.append(line.strip())
    return info


def get_legend(size, classes, colors):
    height, width = size
    legend = np.full((height, width, 3), 255, dtype="uint8")
    colors = np.unique(colors)
    class_height = height // len(colors)

    for i in range(len(colors)):
        _, font_base = cv2.getTextSize(classes[colors[i]].split(',')[0], cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
        cv2.putText(legend, classes[colors[i]].split(',')[0], (110, int((i + 0.625) * class_height)),
                    cv2.FONT_HERSHEY_SIMPLEX, min(1, 0.5*class_height/font_base), (0, 0, 0), 1)
        color = classes_color_map[colors[i]]
        cv2.rectangle(legend, (5, int((i + 0.25) * class_height)), (100, int((i + 0.75) * class_height)),
                      (int(color[0]), int(color[1]), int(color[2])), -1)

    return legend


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    log.info("Creating Inference Engine")
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")
    # Read IR
    log.info("Loading network")
    net = ie.read_network(args.model, os.path.splitext(args.model)[0] + ".bin")

    assert len(net.input_info) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"

    log.info("Preparing input blobs")
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    inputs = get_files(args.input)
    net.batch_size = len(inputs)

    # NB: This is required to load the image as uint8 np.array
    #     Without this step the input blob is loaded in FP32 precision,
    #     this requires additional operation and more memory.
    net.input_info[input_blob].precision = "U8"

    # Read and pre-process input images
    n, c, h, w = net.input_info[input_blob].input_data.shape
    images = np.ndarray(shape=(n, c, h, w))
    for i in range(n):
        image = cv2.imread(inputs[i])
        assert image.dtype == np.uint8
        if image.shape[:-1] != (h, w):
            log.warning("Image {} is resized from {} to {}".format(inputs[i], image.shape[:-1], (h, w)))
            image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        images[i] = image
    log.info("Batch size is {}".format(n))

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)

    # Start sync inference
    log.info("Starting inference")
    res = exec_net.infer(inputs={input_blob: images})

    # Processing output blob
    log.info("Processing output blob")
    res = res[out_blob]
    if len(res.shape) == 3:
        res = np.expand_dims(res, axis=1)
    if len(res.shape) == 4:
        _, _, out_h, out_w = res.shape
    else:
        raise Exception("Unexpected output blob shape {}. Only 4D and 3D output blobs are supported".format(res.shape))

    classes = get_info_from_file(args.labels)
    own_color_map = get_info_from_file(args.colors)
    global classes_color_map
    if own_color_map:
        classes_color_map = [eval(color) for color in own_color_map]

    # Create folder to save results
    os.makedirs(args.output_dir, exist_ok=True)

    for batch, data in enumerate(res):
        if classes:
            classes_map = np.zeros(shape=(out_h, out_w + args.legend_width, 3), dtype=np.int)
        else:
            classes_map = np.zeros(shape=(out_h, out_w, 3), dtype=np.int)
        colors = []
        for i in range(out_h):
            for j in range(out_w):
                if len(data[:, i, j]) == 1:
                    pixel_class = int(data[:, i, j])
                else:
                    pixel_class = np.argmax(data[:, i, j])
                while pixel_class >= len(classes_color_map):
                    new_color = np.random.randint(0, 255, size=3)
                    classes_color_map.append(new_color)
                colors.append(pixel_class)
                classes_map[i, j, :] = classes_color_map[pixel_class]
        if classes:
            legend = get_legend((out_h, args.legend_width), classes, colors)
            classes_map[:, -args.legend_width:, :] = legend

        out_img = os.path.join(args.output_dir, "out_{}.bmp".format(batch))
        cv2.imwrite(out_img, classes_map)
        log.info("Result image was saved to {}".format(out_img))
    log.info("This demo is an API example, for any performance measurements please use the dedicated benchmark_app tool "
             "from the openVINO toolkit\n")


if __name__ == '__main__':
    sys.exit(main() or 0)
