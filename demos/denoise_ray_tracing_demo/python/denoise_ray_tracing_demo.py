#!/usr/bin/env python3

"""
 Copyright (C) 2023 KNS Group LLC (YADRO)
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
from time import perf_counter
import logging as log
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np

from openvino.runtime import Core, get_version
from utils.color import autoexposure, get_transfer_function, round_up, srgb_inverse, srgb_forward

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument("-m", "--model", type=Path, required=True,
                      help="Required. Path to an .xml file with a trained model.")
    args.add_argument("--input_hdr", type=Path, required=True,
                      help="Required. Path to an HDR image to infer")
    args.add_argument("--input_albedo", type=Path, required=True,
                      help="Required. Path to an albedo image to infer")
    args.add_argument("-d", "--device", type=str, default="CPU",
                      help="Optional. Specify the target device to infer on. "
                           "The demo will look for a suitable plugin for device specified. Default value is CPU")
    args.add_argument("--no_show", help="Optional. Don't show output. Cannot be used in GUI mode", action='store_true')
    args.add_argument("-o", "--output", help="Optional. Save output to the file with provided filename.",
                      default="", type=Path)
    args.add_argument('--input_scale', '--is', type=float, default=1.,
                      help='Scales values in the main input image before filtering, '
                           'without scaling the output too')

    return parser


def is_srgb_image(filename):
    return filename.suffix not in ('.pfm', '.phm', '.exr', '.hdr')


def pad_image(image, shape):
    image = np.pad(image, ((0, 0),
                           (0, 0),
                           (0, round_up(shape[2], 16) - shape[2]),
                           (0, round_up(shape[3], 16) - shape[3])))

    return image


def load_image(filename):
    image = cv2.imread(str(filename), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError('Could not read image')

    if is_srgb_image(filename):
        image = srgb_inverse(image)

    image = image[:, :, :3]
    image = np.nan_to_num(image)
    return image


def load_image_features(cfg):
    color = load_image(cfg.input_hdr)
    albedo = load_image(cfg.input_albedo)
    images = {"color": color, "albedo": albedo}
    return images


def preprocess_input(images, args):
    color = images["color"]
    albedo = images["albedo"]

    exposure = autoexposure(color)
    transfer = get_transfer_function()

    if args.input_scale:
        color *= args.input_scale
    color *= exposure

    color = transfer.forward(color)

    # HWC -> BCHW
    color = np.expand_dims(color.transpose((2, 0, 1)), 0)
    shape = color.shape
    color = pad_image(color, shape)

    albedo = np.expand_dims(albedo.transpose((2, 0, 1)), 0)
    albedo = pad_image(albedo, shape)

    params = {"exposure": exposure, "transfer": transfer, "shape": shape}
    features = {"color": color, "albedo": albedo}
    return features, params


def postprocess_image(image, args, params):
    shape = params["shape"]
    image = image[0, :, :shape[2], :shape[3]].transpose((1, 2, 0))
    image = np.maximum(image, 0.)

    image = params.get("transfer").inverse(image)

    image = image / params.get("exposure")

    if is_srgb_image(args.output):
        image = srgb_forward(image)
    return image


def main():
    args = build_argparser().parse_args()

    # Plugin initialization
    log.info('OpenVINO Runtime')
    log.info(f'\tbuild: {get_version()}')
    core = Core()

    if 'GPU' in args.device:
        core.set_property("GPU", {"GPU_ENABLE_LOOP_UNROLLING": "NO", "CACHE_DIR": "./"})

    # Read IR
    log.info(f'Reading model {args.model}')
    model = core.read_model(args.model)

    input_tensor_names = [model.inputs[i].get_any_name() for i in range(len(model.inputs))]

    if len(model.outputs) != 1:
        raise RuntimeError("Demo supports only single output topologies")
    output_tensor_name = model.outputs[0].get_any_name()

    # load input features
    load_start_time = perf_counter()
    images = load_image_features(args)
    load_total_time = perf_counter() - load_start_time

    # pre-process input features
    preprocessing_start_time = perf_counter()
    input_image, params_image = preprocess_input(images, args)
    preprocessing_total_time = perf_counter() - preprocessing_start_time

    # Loading model to the plugin
    compiled_model = core.compile_model(model, args.device)
    infer_request = compiled_model.create_infer_request()
    log.info(f'The model {args.model} is loaded to {args.device}')

    # Start sync inference
    inference_start_time = perf_counter()
    infer_request.infer(inputs={input_tensor_names[0]: input_image[input_tensor_names[0]],
                                input_tensor_names[1]: input_image[input_tensor_names[1]]})
    preds = infer_request.get_tensor(output_tensor_name).data[:]
    inference_total_time = perf_counter() - inference_start_time

    postprocessing_start_time = perf_counter()
    result = postprocess_image(preds, args, params_image)
    postprocessing_total_time = perf_counter() - postprocessing_start_time

    total_latency = (load_total_time
                     + preprocessing_total_time
                     + inference_total_time
                     + postprocessing_total_time) * 1e3

    log.info("Metrics report:")
    log.info(f"\tLatency: {total_latency:.1f} ms")
    log.info(f"\tLoad features: {load_total_time * 1e3:.1f} ms")
    log.info(f"\tPreprocessing: {preprocessing_total_time * 1e3:.1f} ms")
    log.info(f"\tInference: {inference_total_time * 1e3:.1f} ms")
    log.info(f"\tPostprocessing: {postprocessing_total_time * 1e3:.1f} ms")

    if args.output.name != "":
        result_save = result * 255 if is_srgb_image(args.output) else result
        cv2.imwrite(str(args.output), result_save)
    if not args.no_show:
        input_image = images["color"]
        if not is_srgb_image(args.input_hdr):
            input_image = srgb_forward(input_image)
        if not is_srgb_image(args.output):
            result = srgb_forward(result)

        imshow_image = cv2.hconcat([input_image, result])
        cv2.namedWindow("Denoise Ray Tracing Image Demo", cv2.WINDOW_NORMAL)
        cv2.imshow('Denoise Ray Tracing Image Demo', imshow_image)
        cv2.waitKey(0)

    sys.exit()


if __name__ == '__main__':
    main()
