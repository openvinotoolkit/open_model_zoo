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

import numpy as np
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
from PIL import Image
import logging as log
from openvino.inference_engine import IECore
from models import CorrespondenceModel, GenerativeModel, CocosnetModel
from preprocessing import preprocess_with_images, preprocess_with_semantics
from postprocessing import postprocess, save_result

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-c", "--correspondence_model", help="Required. Path to an .xml file with a trained correspondence model",
                      required=True, type=str)
    args.add_argument("-g", "--generative_model", help="Required. Path to an .xml file with a trained generative model",
                      required=True, type=str)
    args.add_argument("-is", "--input_semantics", help="Required. Path to a folder with semantic images or path to a semantic image",
                      required=True, type=str)
    args.add_argument("-ri", "--reference_image", help="Required. Path to a folder with reference images or path to a reference image",
                      required=True, type=str)
    args.add_argument("-rs", "--reference_semantics", help="Required. Path to a folder with reference semantics or path to a reference semantic",
                      required=True, type=str)
    args.add_argument("-o", "--output_dir", help="Required. Path to directory to save the result",
                      required=False, type=str, default="result.jpg")
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. "
                           "Absolute MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the "
                           "kernels implementations", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. Sample will look for a suitable plugin for device specified. Default value is CPU",
                      default="CPU", type=str)
    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    log.info("Creating CoCosNet Model")
    ie_core = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")
    corr_model = CorrespondenceModel(ie_core, args.correspondence_model,
                                     args.correspondence_model.replace(".xml", ".bin"), args.device)
    gen_model = GenerativeModel(ie_core, args.generative_model,
                                args.generative_model.replace(".xml", ".bin"), args.device)
    model = CocosnetModel(corr_model, gen_model)

    log.info("Preparing input data")
    input_semantics = cv2.imread(args.input_semantics, cv2.IMREAD_GRAYSCALE)
    input_semantics = preprocess_with_semantics(input_semantics)
    reference_image = cv2.imread(args.reference_image)
    reference_image = preprocess_with_images(reference_image)
    reference_semantics = cv2.imread(args.reference_semantics, cv2.IMREAD_GRAYSCALE)
    reference_semantics = preprocess_with_semantics(reference_semantics)
    input_data = {
        'input_semantics': input_semantics,
        'reference_image': reference_image,
        'reference_semantics': reference_semantics
    }

    log.info("Inference for input")
    out = model.infer(input_data)
    print("Out = ", out)
    
    log.info("Postprocessing for result")
    result = postprocess(out)
    cv2.imshow("Result", result)
    cv2.waitKey()
    cv2.destroyAllWindows() 
    
    log.info("Save result")
    save_result(out, args.output_dir)
    log.info("Result image was saved to {}".format(args.output_dir))


if __name__ == '__main__':
    sys.exit(main() or 0)