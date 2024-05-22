#!/usr/bin/env python3
"""
 Copyright (c) 2019-2024 Intel Corporation

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

import logging as log
import sys
from pathlib import Path
from time import perf_counter
from argparse import ArgumentParser, SUPPRESS

import cv2
import numpy as np
from scipy.special import softmax
from openvino import Core, get_version

from text_spotting_demo.tracker import StaticIOUTracker

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python/model_zoo'))

import monitors
from images_capture import open_images_capture
from visualizers import InstanceSegmentationVisualizer
from model_api.performance_metrics import PerformanceMetrics

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

SOS_INDEX = 0
EOS_INDEX = 1
MAX_SEQ_LEN = 28


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-m_m', '--mask_rcnn_model',
                      help='Required. Path to an .xml file with a trained Mask-RCNN model with '
                           'additional text features output.',
                      required=True, type=str, metavar='"<path>"')
    args.add_argument('-m_te', '--text_enc_model',
                      help='Required. Path to an .xml file with a trained text recognition model '
                           '(encoder part).',
                      required=True, type=str, metavar='"<path>"')
    args.add_argument('-m_td', '--text_dec_model',
                      help='Required. Path to an .xml file with a trained text recognition model '
                           '(decoder part).',
                      required=True, type=str, metavar='"<path>"')
    args.add_argument('-i', '--input', required=True,
                      help='Required. An input to process. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    args.add_argument('--loop', default=False, action='store_true',
                      help='Optional. Enable reading the input in a loop.')
    args.add_argument('-o', '--output', required=False,
                      help='Optional. Name of the output file(s) to save. Frames of odd width or height can be truncated. See https://github.com/opencv/opencv/pull/24086')
    args.add_argument('-limit', '--output_limit', required=False, default=1000, type=int,
                      help='Optional. Number of frames to store in output. '
                           'If 0 is set, all frames are stored.')
    args.add_argument('-d', '--device',
                      help='Optional. Specify the target device to infer on, i.e : CPU, GPU. '
                           'The demo will look for a suitable plugin for device specified '
                           '(by default, it is CPU). Please refer to OpenVINO documentation '
                           'for the list of devices supported by the model.',
                      default='CPU', type=str, metavar='"<device>"')
    args.add_argument('--delay',
                      help='Optional. Interval in milliseconds of waiting for a key to be pressed.',
                      default=0, type=int, metavar='"<num>"')
    args.add_argument('-pt', '--prob_threshold',
                      help='Optional. Probability threshold for detections filtering.',
                      default=0.5, type=float, metavar='"<num>"')
    args.add_argument('-a', '--alphabet',
                      help='Optional. Alphabet that is used for decoding.',
                      default='  abcdefghijklmnopqrstuvwxyz0123456789')
    args.add_argument('--trd_input_prev_symbol',
                      help='Optional. Name of previous symbol input node to text recognition head decoder part.',
                      default='prev_symbol')
    args.add_argument('--trd_input_prev_hidden',
                      help='Optional. Name of previous hidden input node to text recognition head decoder part.',
                      default='prev_hidden')
    args.add_argument('--trd_input_encoder_outputs',
                      help='Optional. Name of encoder outputs input node to text recognition head decoder part.',
                      default='encoder_outputs')
    args.add_argument('--trd_output_symbols_distr',
                      help='Optional. Name of symbols distribution output node from text recognition head decoder part.',
                      default='output')
    args.add_argument('--trd_output_cur_hidden',
                      help='Optional. Name of current hidden output node from text recognition head decoder part.',
                      default='hidden')
    args.add_argument('-trt', '--tr_threshold',
                      help='Optional. Text recognition confidence threshold.',
                      default=0.5, type=float, metavar='"<num>"')
    args.add_argument('--keep_aspect_ratio',
                      help='Optional. Force image resize to keep aspect ratio.',
                      action='store_true')
    args.add_argument('--no_track',
                      help='Optional. Disable tracking.',
                      action='store_true')
    args.add_argument('--show_scores',
                      help='Optional. Show detection scores.',
                      action='store_true')
    args.add_argument('--show_boxes',
                      help='Optional. Show bounding boxes.',
                      action='store_true')
    args.add_argument('-r', '--raw_output_message',
                      help='Optional. Output inference results raw values.',
                      action='store_true')
    args.add_argument("--no_show",
                      help="Optional. Don't show output",
                      action='store_true')
    args.add_argument('-u', '--utilization_monitors', default='', type=str,
                      help='Optional. List of monitors to show initially.')
    return parser


def expand_box(box, scale):
    w_half = (box[2] - box[0]) * .5
    h_half = (box[3] - box[1]) * .5
    x_c = (box[2] + box[0]) * .5
    y_c = (box[3] + box[1]) * .5
    w_half *= scale
    h_half *= scale
    box_exp = np.zeros(box.shape)
    box_exp[0] = x_c - w_half
    box_exp[2] = x_c + w_half
    box_exp[1] = y_c - h_half
    box_exp[3] = y_c + h_half
    return box_exp


def segm_postprocess(box, raw_cls_mask, im_h, im_w):
    # Add zero border to prevent upsampling artifacts on segment borders.
    raw_cls_mask = np.pad(raw_cls_mask, ((1, 1), (1, 1)), 'constant', constant_values=0)
    extended_box = expand_box(box, raw_cls_mask.shape[0] / (raw_cls_mask.shape[0] - 2.0)).astype(int)
    w, h = np.maximum(extended_box[2:] - extended_box[:2] + 1, 1)
    x0, y0 = np.clip(extended_box[:2], a_min=0, a_max=[im_w, im_h])
    x1, y1 = np.clip(extended_box[2:] + 1, a_min=0, a_max=[im_w, im_h])

    raw_cls_mask = cv2.resize(raw_cls_mask, (w, h)) > 0.5
    mask = raw_cls_mask.astype(np.uint8)
    # Put an object mask in an image mask.
    im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
    im_mask[y0:y1, x0:x1] = mask[(y0 - extended_box[1]):(y1 - extended_box[1]),
                            (x0 - extended_box[0]):(x1 - extended_box[0])]
    return im_mask


def main():
    args = build_argparser().parse_args()

    cap = open_images_capture(args.input, args.loop)

    log.info('OpenVINO Runtime')
    log.info('\tbuild: {}'.format(get_version()))
    core = Core()

    # Read IR
    log.info('Reading Mask-RCNN model {}'.format(args.mask_rcnn_model))
    mask_rcnn_model = core.read_model(args.mask_rcnn_model)

    input_tensor_name = 'image'
    try:
        n, c, h, w = mask_rcnn_model.input(input_tensor_name).shape
        if n != 1:
            raise RuntimeError('Only batch 1 is supported by the demo application')
    except RuntimeError:
        raise RuntimeError('Demo supports only topologies with the following input tensor name: {}'.format(input_tensor_name))

    required_output_names = {'boxes', 'labels', 'masks', 'text_features'}
    for output_tensor_name in required_output_names:
        try:
            mask_rcnn_model.output(output_tensor_name)
        except RuntimeError:
            raise RuntimeError('Demo supports only topologies with the following output tensor names: {}'.format(
                ', '.join(required_output_names)))

    log.info('Reading Text Recognition Encoder model {}'.format(args.text_enc_model))
    text_enc_model = core.read_model(args.text_enc_model)

    log.info('Reading Text Recognition Decoder model {}'.format(args.text_dec_model))
    text_dec_model = core.read_model(args.text_dec_model)

    mask_rcnn_compiled_model = core.compile_model(mask_rcnn_model, device_name=args.device)
    mask_rcnn_infer_request = mask_rcnn_compiled_model.create_infer_request()
    log.info('The Mask-RCNN model {} is loaded to {}'.format(args.mask_rcnn_model, args.device))

    text_enc_compiled_model = core.compile_model(text_enc_model, args.device)
    text_enc_output_tensor = text_enc_compiled_model.outputs[0]
    text_enc_infer_request = text_enc_compiled_model.create_infer_request()
    log.info('The Text Recognition Encoder model {} is loaded to {}'.format(args.text_enc_model, args.device))

    text_dec_compiled_model = core.compile_model(text_dec_model, args.device)
    text_dec_infer_request = text_dec_compiled_model.create_infer_request()
    log.info('The Text Recognition Decoder model {} is loaded to {}'.format(args.text_dec_model, args.device))

    hidden_shape = text_dec_model.input(args.trd_input_prev_hidden).shape
    text_dec_output_names = {args.trd_output_symbols_distr, args.trd_output_cur_hidden}

    if args.no_track:
        tracker = None
    else:
        tracker = StaticIOUTracker()

    if args.delay:
        delay = args.delay
    else:
        delay = int(cap.get_type() in ('VIDEO', 'CAMERA'))

    visualizer = InstanceSegmentationVisualizer(show_boxes=args.show_boxes, show_scores=args.show_scores)

    frames_processed = 0

    metrics = PerformanceMetrics()
    video_writer = cv2.VideoWriter()

    start_time = perf_counter()
    frame = cap.read()
    if frame is None:
        raise RuntimeError("Can't read an image from the input")

    presenter = monitors.Presenter(args.utilization_monitors, 45, (frame.shape[1] // 4, frame.shape[0] // 8))
    if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'),
                                             cap.fps(), (frame.shape[1], frame.shape[0])):
        raise RuntimeError("Can't open video writer")

    while frame is not None:
        if not args.keep_aspect_ratio:
            # Resize the image to a target size.
            scale_x = w / frame.shape[1]
            scale_y = h / frame.shape[0]
            input_image = cv2.resize(frame, (w, h))
        else:
            # Resize the image to keep the same aspect ratio and to fit it to a window of a target size.
            scale_x = scale_y = min(h / frame.shape[0], w / frame.shape[1])
            input_image = cv2.resize(frame, None, fx=scale_x, fy=scale_y)

        input_image_size = input_image.shape[:2]
        input_image = np.pad(input_image, ((0, h - input_image_size[0]),
                                           (0, w - input_image_size[1]),
                                           (0, 0)),
                             mode='constant', constant_values=0)
        # Change data layout from HWC to CHW.
        input_image = input_image.transpose((2, 0, 1))
        input_image = input_image.reshape((n, c, h, w)).astype(np.float32)

        # Run the MaskRCNN model.
        mask_rcnn_infer_request.infer({input_tensor_name: input_image})
        outputs = {name: mask_rcnn_infer_request.get_tensor(name).data[:] for name in required_output_names}

        # Parse detection results of the current request
        boxes = outputs['boxes'][:, :4]
        scores = outputs['boxes'][:, 4]
        classes = outputs['labels'].astype(np.uint32)
        raw_masks = outputs['masks']
        text_features = outputs['text_features']

        # Filter out detections with low confidence.
        detections_filter = scores > args.prob_threshold
        scores = scores[detections_filter]
        classes = classes[detections_filter]
        boxes = boxes[detections_filter]
        raw_masks = raw_masks[detections_filter]
        text_features = text_features[detections_filter]

        boxes[:, 0::2] /= scale_x
        boxes[:, 1::2] /= scale_y
        masks = []
        for box, cls, raw_mask in zip(boxes, classes, raw_masks):
            mask = segm_postprocess(box, raw_mask, frame.shape[0], frame.shape[1])
            masks.append(mask)

        texts = []
        for feature in text_features:
            input_data = {'input': np.expand_dims(feature, axis=0)}
            feature = text_enc_infer_request.infer(input_data)[text_enc_output_tensor]
            feature = np.reshape(feature, (feature.shape[0], feature.shape[1], -1))
            feature = np.transpose(feature, (0, 2, 1))

            hidden = np.zeros(hidden_shape)
            prev_symbol_index = np.ones((1,)) * SOS_INDEX

            text = ''
            text_confidence = 1.0
            for i in range(MAX_SEQ_LEN):
                text_dec_infer_request.infer({
                    args.trd_input_prev_symbol: np.reshape(prev_symbol_index, (1,)),
                    args.trd_input_prev_hidden: hidden,
                    args.trd_input_encoder_outputs: feature})
                decoder_output = {name: text_dec_infer_request.get_tensor(name).data[:] for name in text_dec_output_names}
                symbols_distr = decoder_output[args.trd_output_symbols_distr]
                symbols_distr_softmaxed = softmax(symbols_distr, axis=1)[0]
                prev_symbol_index = int(np.argmax(symbols_distr, axis=1))
                text_confidence *= symbols_distr_softmaxed[prev_symbol_index]
                if prev_symbol_index == EOS_INDEX:
                    break
                text += args.alphabet[prev_symbol_index]
                hidden = decoder_output[args.trd_output_cur_hidden]

            texts.append(text if text_confidence >= args.tr_threshold else '')

        if len(boxes) and args.raw_output_message:
            log.debug('  -------------------------- Frame # {} --------------------------  '.format(frames_processed))
            log.debug('  Class ID | Confidence |     XMIN |     YMIN |     XMAX |     YMAX ')
            for box, cls, score, mask in zip(boxes, classes, scores, masks):
                log.debug('{:>10} | {:>10f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} '.format(cls, score, *box))

        # Get instance track IDs.
        masks_tracks_ids = None
        if tracker is not None:
            masks_tracks_ids = tracker(masks, classes)

        presenter.drawGraphs(frame)

        # Visualize masks.
        frame = visualizer(frame, boxes, classes, scores, masks, masks_tracks_ids, texts)
        metrics.update(start_time, frame)

        frames_processed += 1
        if video_writer.isOpened() and (args.output_limit <= 0 or frames_processed <= args.output_limit):
            video_writer.write(frame)

        if not args.no_show:
            # Show resulting image.
            cv2.imshow('Results', frame)

        if not args.no_show:
            key = cv2.waitKey(delay)
            esc_code = 27
            if key == esc_code:
                break
            presenter.handleKey(key)

        start_time = perf_counter()
        frame = cap.read()

    metrics.log_total()
    for rep in presenter.reportMeans():
        log.info(rep)


if __name__ == '__main__':
    sys.exit(main() or 0)
