"""
 Copyright (C) 2020 Intel Corporation
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
from models import CorrespondenceModel, GenerativeModel, CocosnetModel, SegmentationModel
from preprocessing import preprocess_with_images, preprocess_with_semantics, preprocess_for_seg_model
from postprocessing import postprocess, save_result


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument("-c", "--correspondence_model",
                      help="Required. Path to an .xml file with a trained correspondence model",
                      required=True, type=str)
    args.add_argument("-g", "--generative_model",
                      help="Required. Path to an .xml file with a trained generative model",
                      required=True, type=str)
    args.add_argument("-s", "--segmentation_model",
                      help="Required. Path to an .xml file with a trained semantic segmentation model",
                      required=False, default=None, type=str)
    args.add_argument("-ii", "--input_images",
                      help="Required. Path to a folder with input images or path to a input image",
                      default="", type=str)
    args.add_argument("-is", "--input_semantics",
                      help="Required. Path to a folder with semantic images or path to a semantic image",
                      default="", type=str)
    args.add_argument("-ri", "--reference_images",
                      help="Required. Path to a folder with reference images or path to a reference image",
                      default="", type=str)
    args.add_argument("-rs", "--reference_semantics",
                      help="Required. Path to a folder with reference semantics or path to a reference semantic",
                      default="", type=str)
    args.add_argument("-o", "--output_dir", help="Path to directory to save the results",
                      required=False, type=str, default="results")
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. "
                           "Absolute MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the "
                           "kernels implementations", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. Sample will look for a suitable plugin for device specified. Default value is CPU",
                      default="CPU", type=str)
    return parser


def get_files(path):
    if os.path.isdir(path):
        return os.listdir(path)
    return [""]


def get_mask_from_image(image, model):
    image = preprocess_for_seg_model(image)
    res = model.infer(image)
    mask = np.argmax(res, axis=1)
    mask = np.squeeze(mask, 0)
    return mask + 1


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    log.info("Creating CoCosNet Model")
    ie_core = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie_core.add_extension(args.cpu_extension, "CPU")
    corr_model = CorrespondenceModel(ie_core, args.correspondence_model,
                                     args.correspondence_model.replace(".xml", ".bin"),
                                     args.device)
    gen_model = GenerativeModel(ie_core, args.generative_model,
                                args.generative_model.replace(".xml", ".bin"),
                                args.device)
    seg_model = SegmentationModel(ie_core, args.segmentation_model,
                                  args.segmentation_model.replace(".xml", ".bin"),
                                  args.device) if args.segmentation_model else None
    model = CocosnetModel(corr_model, gen_model)

    log.info("Preparing input data")
    input_data = []
    assert args.reference_images and ((args.input_semantics and
           args.reference_semantics) or args.input_images), "Not enough data to do inference"
    input_images = sorted(get_files(args.input_images))
    input_semantics = sorted(get_files(args.input_semantics))
    reference_images = sorted(get_files(args.reference_images))
    reference_semantics = sorted(get_files(args.reference_semantics))
    n = len(reference_images)
    if seg_model:
        players = [input_images, n * [''], reference_images, n * ['']]
    else:
        players = [n * [''], input_semantics, reference_images, reference_semantics]
    for input_img, input_sem, ref_img, ref_sem in zip(*players):
        if seg_model:
            input_sem = get_mask_from_image(cv2.imread(args.input_images + input_img), seg_model)
            ref_sem = get_mask_from_image(cv2.imread(args.reference_images + ref_img), seg_model)
        else:
            input_sem = cv2.imread(args.input_semantics + input_sem, cv2.IMREAD_GRAYSCALE)
            ref_sem = cv2.imread(args.reference_semantics + ref_sem, cv2.IMREAD_GRAYSCALE)
        input_sem = preprocess_with_semantics(input_sem)
        ref_img = preprocess_with_images(cv2.imread(args.reference_images + ref_img))
        ref_sem = preprocess_with_semantics(ref_sem)
        input_dict = {
            'input_semantics': input_sem,
            'reference_image': ref_img,
            'reference_semantics': ref_sem
        }
        input_data.append(input_dict)

    log.info("Inference for input")
    outs = []
    for data in input_data:
        outs.append(model.infer(data))

    log.info("Postprocessing for result")
    results = [postprocess(out) for out in outs]

    if args.output_dir:
        save_result(outs, args.output_dir)
    log.info("Result image was saved to {}".format(args.output_dir))


if __name__ == '__main__':
    sys.exit(main() or 0)
