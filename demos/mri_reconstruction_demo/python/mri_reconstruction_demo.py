# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import argparse
import time
import logging
import sys

import numpy as np
import cv2 as cv
from openvino import Core

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger('mri_reconstruction_demo')

def kspace_to_image(kspace):
    assert len(kspace.shape) == 3 and kspace.shape[-1] == 2
    fft = cv.idft(kspace, flags=cv.DFT_SCALE)
    img = cv.magnitude(fft[:, :, 0], fft[:, :, 1])
    return cv.normalize(img, dst=None, alpha=255, beta=0, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)


def build_argparser():
    parser = argparse.ArgumentParser(description='MRI reconstrution demo')
    parser.add_argument('-i', '--input', dest='input', required=True,
                        help='Path to input .npy file with MRI scan data.')
    parser.add_argument('-p', '--pattern', dest='pattern', required=True,
                        help='Path to sampling mask in .npy format.')
    parser.add_argument('-m', '--model', dest='model', required=True,
                        help='Path to .xml file of OpenVINO IR.')
    parser.add_argument('-d', '--device', dest='device', default='CPU',
                        help='Optional. Specify the target device to infer on; CPU or '
                             'GPU is acceptable. Default value is CPU.')
    parser.add_argument('--no_show', action='store_true',
                        help='Disable results visualization')
    return parser

def main():
    args = build_argparser().parse_args()

    core = Core()
    model = core.read_model(args.model)

    input_tensor_name = 'input_1:0'
    output_candidates = [node.get_any_name() for node in model.outputs if node.shape[3] == 1]
    if len(output_candidates) != 1:
        raise RuntimeError("The model expects output tensor with 1 channel")
    output_tensor_name = output_candidates[0]

    compiled_model = core.compile_model(model, args.device)
    infer_request = compiled_model.create_infer_request()

    # Hybrid-CS-Model-MRI/Data/stats_fs_unet_norm_20.npy
    stats = np.array([2.20295299e-01, 1.11048916e+03], dtype=np.float32)
    # Hybrid-CS-Model-MRI/Data/sampling_mask_20perc.npy
    var_sampling_mask = np.load(args.pattern)
    logger.info(f'Sampling ratio: {1.0 - var_sampling_mask.sum() / var_sampling_mask.size}')

    data = np.load(args.input)
    num_slices, height, width = data.shape[0], data.shape[1], data.shape[2]
    pred = np.zeros((num_slices, height, width), dtype=np.uint8)
    data /= np.sqrt(height * width)

    logger.info('Compute...')
    start = time.time()
    for slice_id, kspace in enumerate(data):
        kspace = kspace.copy()

        # Apply sampling
        kspace[var_sampling_mask] = 0
        kspace = (kspace - stats[0]) / stats[1]

        # Forward through network
        input = np.expand_dims(kspace.transpose(2, 0, 1), axis=0)
        infer_request.infer(inputs={input_tensor_name: input})
        output = infer_request.get_tensor(output_tensor_name).data[:]
        output = output.reshape(height, width)

        # Save predictions
        pred[slice_id] = cv.normalize(output, dst=None, alpha=255, beta=0, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

    logger.info('Elapsed time: %.1f seconds' % (time.time() - start))

    WIN_NAME = 'MRI reconstruction with OpenVINO'

    def callback(slice_id):
        kspace = data[slice_id]
        img = kspace_to_image(kspace)

        kspace[var_sampling_mask] = 0
        masked = kspace_to_image(kspace)

        rec = pred[slice_id]

        # Add a header
        border_size = 20
        render = cv.hconcat((img, masked, rec))
        render = cv.copyMakeBorder(render, border_size, 0, 0, 0, cv.BORDER_CONSTANT, value=255)
        cv.putText(render, 'Original', (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, color=0)
        cv.putText(render, 'Sampled (PSNR %.1f)' % cv.PSNR(img, masked), (width, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, color=0)
        cv.putText(render, 'Reconstructed (PSNR %.1f)' % cv.PSNR(img, rec), (width*2, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, color=0)

        cv.imshow(WIN_NAME, render)
        cv.waitKey(1)

    if not args.no_show:
        cv.namedWindow(WIN_NAME, cv.WINDOW_AUTOSIZE)
        cv.createTrackbar('Slice', WIN_NAME, num_slices // 2, num_slices - 1, callback)
        callback(num_slices // 2)  # Trigger initial visualization
        cv.waitKey()

if __name__ == '__main__':
    sys.exit(main() or 0)
