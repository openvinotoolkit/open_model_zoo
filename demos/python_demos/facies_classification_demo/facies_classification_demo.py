#!/usr/bin/env python3
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

import logging
import os
import sys
from argparse import ArgumentParser, SUPPRESS

import matplotlib.pyplot as plt

import numpy as np

from openvino.inference_engine import IECore


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-m', '--model',
                      help='Required. \
                      Path to an .xml file with a trained model.',
                      required=True, type=str)
    args.add_argument('-i', '--data_path',
                      help='Required. Path to seismic datafile.',
                      required=True, type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Required. Required for CPU custom layers. \
                      (MaxUnpool layers in this case) \
                      Absolute MKLDNN (CPU)-targeted custom layers. "
                      "Absolute path to a shared library \
                      with the kernels implementations",
                      required=True, type=str)
    args.add_argument('-t', '--slice_type',
                      help='Type of slice .',
                      choices=['inline', 'crossline', 'timeline'],
                      type=str, default='crossline')
    args.add_argument('-s', '--slice_index',
                      help='Index of slice .',
                      type=int, default=0)
    args.add_argument('-d', '--device',
                      help='Optional. Specify the target device to infer on: \
                      CPU, GPU, FPGA, HDDL or MYRIAD. '
                      'The demo will look for a suitable plugin \
                      for device specified '
                      '(by default, it is CPU).',
                      default='CPU', type=str, metavar='"<device>"')
    return parser


def get_logger():
    ''' Setting up Logging and return logger.'''
    INFO, INFER = 5, 6
    logging.addLevelName(INFO, 'INFO')

    def info(self, message, *args, **kws):
        self.log(INFO, message, *args, **kws)
    logging.Logger.info = info

    logging.addLevelName(INFER, 'INFER')

    def infer(self, message, *args, **kws):
        self.log(INFER, message, *args, **kws)
    logging.Logger.infer = infer

    logging.basicConfig(format='[%(levelname)s] %(message)s',
                        stream=sys.stdout)
    logger = logging.getLogger('conv_deconvnet_demo')
    logger.setLevel(INFO)
    return logger


def normalize(data, mu=0, std=1):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    data = (data - data.flatten().mean())/data.flatten().std()
    return data * std + mu


def load_data(data_path, slice_index, slice_type):
    data_format = data_path.split('.')[1]
    assert not (data_path.split('.')[0] == '' or data_format == ''), \
        f'Invalid path to data file: {data_path}'
    if data_format == 'npy':
        data = np.load(data_path)
    elif data_format == 'dat':
        data = np.fromfile(data_path)
    elif data_format == 'segy':
        import segyio
        data = segyio.tools.cube(data_path)
        data = np.moveaxis(data, -1, 0)
        data = np.ascontiguousarray(data, 'float32')
    else:
        assert False, f'Unsupported data format: {data_format}'

    data = normalize(data, mu=1e-8, std=0.2097654)
    print(f"[INFO] Dataset has been loaded, shape is {data.shape}")
    print(f"[INFO] Dataset mean is {data.flatten().mean():.5f}, std {data.flatten().std():.5f}")

    assert slice_index >= 0, \
        'Invalid slice index argument, slice index must not be negative'
    x, y, z = data.shape
    if slice_type == 'inline':
        assert slice_index < x, f'Invalid slice index, must be in {[0, x - 1]}'
        x_clice = data[slice_index, :, :]
    elif slice_type == 'crossline':
        assert slice_index < y, f'Invalid slice index, must be in {[0, y - 1]}'
        x_clice = data[:, slice_index, :]
    elif slice_type == 'timeline':
        assert slice_index < z, f'Invalid slice index, must be in {[0, z - 1]}'
        x_clice = data[:, :, slice_index]
    return x_clice.T


def decode_segmap(label_mask):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
            the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
            in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    label_colours = np.asarray([[69, 117, 180], [145, 191, 219],
                                [224, 243, 248], [254, 224, 144],
                                [252, 141, 89], [215, 48, 39]])
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, len(label_colours)):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb


def show_interpretation(input_to_net, out_from_net):
    from matplotlib.colors import LinearSegmentedColormap

    res = np.argmax(out_from_net, axis=1).squeeze()
    res_image = decode_segmap(res)

    color_list = np.asarray([[69, 117, 180], [145, 191, 219],
                            [224, 243, 248], [254, 224, 144],
                            [252, 141, 89], [215, 48, 39]]) / 255
    cm = LinearSegmentedColormap.from_list('custom_cmap', color_list, N=6)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Facies classification results", fontsize=22)
    axs[0].imshow(input_to_net, cmap='gray')
    axs[0].set_title('Input slice')

    im = axs[1].imshow(res_image, cmap=cm)
    axs[1].set_title('Interpretation of the slice')

    cbaxes = fig.add_axes([0.91, 0.2, 0.01, 0.6])
    cb = fig.colorbar(im, ax=axs[1], cax=cbaxes,
                      ticks=[0.33, 0.45, 0.57, 0.69, 0.82, 0.93])
    cb.ax.set_yticklabels(['upper_ns', 'middle_ns', 'lower_ns',
                          'rijnland_chalk', 'scruff', 'zechstein'],
                          fontsize=9, ha="left")
    plt.savefig('interpretation.png')


def main(args, logger):
    logger.info("Create inference engine")
    ie = IECore()

    if args.cpu_extension and "CPU" in args.device:
        logger.info(f'Using custom user CPU extention: {args.cpu_extension}')
        ie.add_extension(args.cpu_extension, "CPU")

    logger.info("Loading network")
    net = ie.read_network(args.model, os.path.splitext(args.model)[0] + ".bin")

    logger.info("Loading dataset")
    inp = load_data(args.data_path, args.slice_index, args.slice_type)

    input_net_info = net.input_info
    input_name = next(iter(input_net_info))
    input_net_shape = input_net_info[input_name].input_data.shape

    if inp.shape != input_net_shape[-2:]:
        logger.info(f"Reshaping model to fit for given slice shape: {inp.shape}")
        net.reshape({input_name: [1, 1, *inp.shape]})

    logger.info("loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)

    logger.infer("Starting inference")
    out = exec_net.infer(inputs={'input': inp})['output']

    logger.info("Showing results")
    show_interpretation(inp, out)


if __name__ == '__main__':
    args = build_argparser().parse_args()
    logger = get_logger()
    main(args, logger)
