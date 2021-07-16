#!/usr/bin/env python3
"""
 Copyright (c) 2019 Intel Corporation

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

import os
import sys
import logging as log

import numpy as np
import nibabel as nib

from PIL import Image, ImageSequence
from datetime import datetime
from argparse import ArgumentParser, SUPPRESS
from fnmatch import fnmatch
from scipy.ndimage import interpolation

from openvino.inference_engine import IECore, get_version


log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

CLASSES_COLOR_MAP = [
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

# suffixes for original interpretation in dataset
SUFFIX_T1 = "_t1.nii.gz"
SUFFIX_T2 = "_t2.nii.gz"
SUFFIX_FLAIR = "_flair.nii.gz"
SUFFIX_T1CE = "_t1ce.nii.gz"
SUFFIX_SEG = "_seg.nii.gz"
# file suffixes to form a data tensor
DATA_SUFFIXES = [SUFFIX_T1, SUFFIX_T2, SUFFIX_FLAIR, SUFFIX_T1CE]

NIFTI_FOLDER = 0
NIFTI_FILE = 1
TIFF_FILE = 2


def mri_sequence(arg):
    sequence = tuple(int(k) for k in arg.split(','))
    if len(sequence) != 4:
        raise AttributeError("The MRI-sequence should contain exactly 4 values, but contains {}.".format(len(sequence)))
    if len(set(sequence)) != 4:
        raise AttributeError("The MRI-sequence has repeating scan types - {}. "
                             "The MRI-sequence must contain native T1, native T2, T2-FLAIR, "
                             "post-Gadolinium contrast T1 scans in the specific for the net order".
                             format(sequence))
    return sequence


def parse_arguments():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-i', '--path_to_input_data', type=str, required=True,
                        help="Required. Path to an input folder with NIfTI data/NIFTI file/TIFF file")
    args.add_argument('-m', '--path_to_model', type=str, required=True,
                        help="Required. Path to an .xml file with a trained model")
    args.add_argument('-o', '--path_to_output', type=str, required=True,
                        help="Required. Path to a folder where output files will be saved")
    args.add_argument('-d', '--target_device', type=str, required=False, default="CPU",
                        help="Optional. Specify a target device to infer on: CPU, GPU. "
                             "Use \"-d HETERO:<comma separated devices list>\" format to specify HETERO plugin.")
    args.add_argument('-l', '--path_to_extension', type=str, required=False, default=None,
                        help="Required for CPU custom layers. "
                             "Absolute path to a shared library with the kernels implementations.")
    args.add_argument("-nii", "--output_nifti", help="Show output inference results as raw values", default=False,
                        action="store_true")
    args.add_argument('-nthreads', '--number_threads', type=int, required=False, default=None,
                        help="Optional. Number of threads to use for inference on CPU (including HETERO cases).")
    args.add_argument('-s', '--shape', nargs='*', type=int, required=False, default=None,
                        help="Optional. Specify shape for a network")
    args.add_argument('-c', '--path_to_cldnn_config', type=str, required=False,
                        help="Required for GPU custom kernels. "
                             "Absolute path to an .xml file with the kernels description.")
    args.add_argument('-ms', '--mri_sequence', type=mri_sequence, metavar='N1,N2,N3,N4', default=(0, 1, 2, 3),
                      help='Optional. Transfer MRI-sequence from dataset order to the network order.')
    args.add_argument("--full_intensities_range", required=False, default=False, action="store_true",
                      help="Take intensities of the input image in a full range.")
    return parser.parse_args()


def get_input_type(path):
    if os.path.isdir(path):
        return NIFTI_FOLDER
    elif fnmatch(path, '*.nii.gz') or fnmatch(path, '*.nii'):
        return NIFTI_FILE
    elif fnmatch(path, '*.tif') or fnmatch(path, '*.tiff'):
        return TIFF_FILE

    raise RuntimeError("Input must be a folder with 4 NIFTI files, single NIFTI file (*.nii or *.nii.gz) or "
                         "TIFF file (*.tif or *.tiff)")


def find_series_name(path):
    for file in os.listdir(path):
        if fnmatch(file, '*.nii.gz'):
            for suffix in DATA_SUFFIXES:
                if suffix in file:
                    return file.replace(suffix, '')


def bbox3(img):
    rows = np.any(img, axis=1)
    rows = np.any(rows, axis=1)
    rows = np.where(rows)

    cols = np.any(img, axis=0)
    cols = np.any(cols, axis=1)
    cols = np.where(cols)

    slices = np.any(img, axis=0)
    slices = np.any(slices, axis=0)
    slices = np.where(slices)

    if (rows[0].shape[0] > 0):
        rmin, rmax = rows[0][[0, -1]]
        cmin, cmax = cols[0][[0, -1]]
        smin, smax = slices[0][[0, -1]]

        return np.array([[rmin, cmin, smin], [rmax, cmax, smax]])
    return np.array([[-1, -1, -1], [0, 0, 0]])


def read_nii_header(data_path, name):
    filename = os.path.join(data_path, name)
    if not os.path.exists(filename):
        raise ValueError("File {} is not exist. Please, validate path to input".format(filename))
    return nib.load(filename)


def normalize(image, mask, full_intensities_range):
    ret = image.copy()
    image_masked = np.ma.masked_array(ret, ~(mask))
    ret = ret - np.mean(image_masked)
    ret = ret / np.var(image_masked) ** 0.5
    if not full_intensities_range:
        ret[ret > 5.] = 5.
        ret[ret < -5.] = -5.
        ret += 5.
        ret /= 10
        ret[~mask] = 0.
    return ret


def resample_np(data, output_shape, order):
    assert(len(data.shape) == len(output_shape))
    factor = [float(o) / i for i, o in zip(data.shape, output_shape)]
    return interpolation.zoom(data, zoom=factor, order=order)


def read_image(test_data_path, data_name, sizes=(128, 128, 128), is_series=True,
               mri_sequence_order=(0, 1, 2, 3), full_intensities_range=False):
    images_list = []
    original_shape = ()
    bboxes = np.zeros(shape=(len(DATA_SUFFIXES),) + (2, 3))

    if is_series:
        data_seq = [DATA_SUFFIXES[i] for i in mri_sequence_order]
        for j, s in enumerate(data_seq):
            image_handle = read_nii_header(test_data_path, data_name + s)
            affine = image_handle.affine
            image = image_handle.get_data().astype(np.float32)

            mask = image > 0.
            bboxes[j] = bbox3(mask)
            image = normalize(image, mask, full_intensities_range)

            images_list.append(image.reshape((1, 1,) + image.shape))
            original_shape = image.shape
    else:
        data_handle = read_nii_header(test_data_path, data_name)
        affine = data_handle.affine
        data = data_handle.get_data().astype(np.float32)
        assert len(data.shape) == 4, 'Wrong data dimensions - {}, must be 4'.format(len(data.shape))
        assert data.shape[3] == 4, 'Wrong data shape - {}, must be (:,:,:,4)'.format(data.shape)
        # Reading order is specified for data from http://medicaldecathlon.com/
        for j in mri_sequence_order:
            image = data[:, :, :, j]
            mask = image > 0
            bboxes[j] = bbox3(mask)
            image = normalize(image, mask, full_intensities_range)
            images_list.append(image.reshape((1, 1,) + image.shape))
        original_shape = data.shape[:3]

    bbox_min = np.min(bboxes[:, 0, :], axis=0).ravel().astype(int)
    bbox_max = np.max(bboxes[:, 1, :], axis=0).ravel().astype(int)
    bbox = np.zeros(shape=(2, 3), dtype=np.float)
    bbox[0] = bbox_min
    bbox[1] = bbox_max

    data = np.concatenate(images_list, axis=1)
    if data.shape[2:] == sizes:
        data_crop = data
    else:
        data_crop = resample_np(
            data[:, :, bbox_min[0]:bbox_max[0], bbox_min[1]:bbox_max[1], bbox_min[2]:bbox_max[2]],
            (1, len(DATA_SUFFIXES),) + sizes,
            1)

    bbox_ret = [
        bbox_min[0], bbox_max[0],
        bbox_min[1], bbox_max[1],
        bbox_min[2], bbox_max[2]
    ]

    return data, data_crop, affine, original_shape, bbox_ret


def main():
    args = parse_arguments()

    # --------------------------------- 1. Load Plugin for inference engine ---------------------------------
    log.info('OpenVINO Inference Engine')
    log.info('\tbuild: {}'.format(get_version()))
    ie = IECore()

    if 'CPU' in args.target_device:
        if args.path_to_extension:
            ie.add_extension(args.path_to_extension, "CPU")
        if args.number_threads is not None:
            ie.set_config({'CPU_THREADS_NUM': str(args.number_threads)}, "CPU")
    elif 'GPU' in args.target_device:
        if args.path_to_cldnn_config:
            ie.set_config({'CONFIG_FILE': args.path_to_cldnn_config}, "GPU")
    else:
        raise AttributeError("Device {} do not support of 3D convolution. "
                             "Please use CPU, GPU or HETERO:*CPU*, HETERO:*GPU*")

    # --------------------- 2. Read IR Generated by ModelOptimizer (.xml and .bin files) ---------------------
    log.info('Reading model {}'.format(args.path_to_model))
    ie_network = ie.read_network(args.path_to_model, os.path.splitext(args.path_to_model)[0] + '.bin')

    input_info = ie_network.input_info
    if len(input_info) == 0:
        raise AttributeError('No inputs info is provided')
    elif len(input_info) != 1:
        raise AttributeError("only one input layer network is supported")

    input_name = next(iter(input_info))
    out_name = next(iter(ie_network.outputs))

    if args.shape:
        log.debug("Reshape model from {} to {}".format(input_info[input_name].input_data.shape, args.shape))
        ie_network.reshape({input_name: args.shape})
        input_info = ie_network.input_info

    # ---------------------------------------- 4. Preparing input data ----------------------------------------

    if len(input_info[input_name].input_data.shape) != 5:
        raise AttributeError("Incorrect shape {} for 3d convolution network".format(args.shape))

    n, c, d, h, w = input_info[input_name].input_data.shape
    ie_network.batch_size = n

    if not os.path.exists(args.path_to_input_data):
        raise AttributeError("Path to input data: '{}' does not exist".format(args.path_to_input_data))

    input_type = get_input_type(args.path_to_input_data)
    is_nifti_data = (input_type == NIFTI_FILE or input_type == NIFTI_FOLDER)

    if input_type == NIFTI_FOLDER:
        series_name = find_series_name(args.path_to_input_data)
        original_data, data_crop, affine, original_size, bbox = \
            read_image(args.path_to_input_data, data_name=series_name, sizes=(d, h, w),
                       mri_sequence_order=args.mri_sequence, full_intensities_range=args.full_intensities_range)

    elif input_type == NIFTI_FILE:
        original_data, data_crop, affine, original_size, bbox = \
            read_image(args.path_to_input_data, data_name=args.path_to_input_data, sizes=(d, h, w), is_series=False,
                       mri_sequence_order=args.mri_sequence, full_intensities_range=args.full_intensities_range)
    else:
        data_crop = np.zeros(shape=(n, c, d, h, w), dtype=np.float)
        im_seq = ImageSequence.Iterator(Image.open(args.path_to_input_data))
        for i, page in enumerate(im_seq):
            im = np.array(page).reshape(h, w, c)
            for channel in range(c):
                data_crop[:, channel, i, :, :] = im[:, :, channel]
        original_data = data_crop
        original_size = original_data.shape[-3:]

    test_im = {input_name: data_crop}

    # ------------------------------------- 4. Loading model to the plugin -------------------------------------
    executable_network = ie.load_network(network=ie_network, device_name=args.target_device)
    log.info('The model {} is loaded to {}'.format(args.path_to_model, args.target_device))
    del ie_network

    # ---------------------------------------------- 5. Do inference --------------------------------------------
    start_time = datetime.now()
    res = executable_network.infer(test_im)
    infer_time = datetime.now() - start_time
    log.info("Inference time is {}".format(infer_time))

    # ---------------------------- 6. Processing of the received inference results ------------------------------
    result = res[out_name]
    batch, channels, out_d, out_h, out_w = result.shape

    list_img = []
    list_seg_result = []

    start_time = datetime.now()
    for batch, data in enumerate(result):
        seg_result = np.zeros(shape=original_size, dtype=np.uint8)
        if data.shape[1:] != original_size:
            x = bbox[1] - bbox[0]
            y = bbox[3] - bbox[2]
            z = bbox[5] - bbox[4]
            out_result = np.zeros(shape=((channels,) + original_size), dtype=float)
            out_result[:, bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]] = \
                resample_np(data, (channels, x, y, z), 1)
        else:
            out_result = data

        if channels == 1:
            reshaped_data = out_result.reshape(original_size[0], original_size[1], original_size[2])
            mask = reshaped_data[:, :, :] > 0.5
            reshaped_data[mask] = 1
            seg_result = reshaped_data.astype(int)
        elif channels == 4:
            seg_result = np.argmax(out_result, axis=0).astype(int)
        elif channels == 3:
            res = np.zeros(shape=out_result.shape, dtype=bool)
            res = out_result > 0.5
            wt = res[0]
            tc = res[1]
            et = res[2]

            seg_result[wt] = 2
            seg_result[tc] = 1
            seg_result[et] = 3

        im = np.stack([original_data[batch, 0, :, :, :],
                       original_data[batch, 0, :, :, :],
                       original_data[batch, 0, :, :, :]],
                      axis=3)

        im = 255 * (im - im.min())/(im.max() - im.min())
        color_seg_frame = np.zeros(im.shape, dtype=np.uint8)
        for idx, c in enumerate(CLASSES_COLOR_MAP):
            color_seg_frame[seg_result[:, :, :] == idx, :] = np.array(c, dtype=np.uint8)
        mask = seg_result[:, :, :] > 0
        im[mask] = color_seg_frame[mask]

        for k in range(im.shape[2]):
            if is_nifti_data:
                list_img.append(Image.fromarray(im[:, :, k, :].astype('uint8'), 'RGB'))
            else:
                list_img.append(Image.fromarray(im[k, :, :, :].astype('uint8'), 'RGB'))

        if args.output_nifti and is_nifti_data:
            list_seg_result.append(seg_result)

    result_processing_time = datetime.now() - start_time
    log.info("Processing time is {}".format(result_processing_time))

    # --------------------------------------------- 7. Save output -----------------------------------------------
    tiff_output_name = os.path.join(args.path_to_output, 'output.tiff')
    Image.new('RGB', (original_data.shape[3], original_data.shape[2])).save(tiff_output_name,
        append_images=list_img, save_all=True)
    log.debug("Result tiff file was saved to {}".format(tiff_output_name))

    if args.output_nifti and is_nifti_data:
        for seg_res in list_seg_result:
            nii_filename = os.path.join(args.path_to_output, 'output_{}.nii.gz'.format(list_seg_result.index(seg_res)))
            nib.save(nib.Nifti1Image(seg_res, affine=affine), nii_filename)
            log.debug("Result nifti file was saved to {}".format(nii_filename))

if __name__ == "__main__":
    main()
