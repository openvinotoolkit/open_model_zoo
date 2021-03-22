#!/usr/bin/env python3
"""
 Copyright (C) 2018-2021 Intel Corporation

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
import argparse
from math import pi, ceil
import time
import logging
from pathlib import Path
from time import perf_counter

import numpy as np
import cv2 as cv
from openvino.inference_engine import IECore

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))

from pipelines import AsyncPipeline

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()

class OpenSlideImageReader:
    def __init__(self, path, pixel_size=0.5):
        from openslide import OpenSlide
        self.img = OpenSlide(path)

        # Get level for a given scale
        level = None
        for i, scale in enumerate(self.img.level_downsamples):
            if pixel_size == 1.0 / scale:
                level = i
                break
        if not level:
            raise Exception('No level found for pixel size {}. Available levels are {}'.format(pixel_size, self.img.level_dimensions))

        self.width = self.img.level_dimensions[level][0]
        self.height = self.img.level_dimensions[level][1]
        self.level = level
        self.scale = int(self.img.level_downsamples[level])


    def get_tile(self, tile_id, sz):
        x, y = tile_id
        y = min(y * self.scale * sz, self.height - sz)  # Last tile offset
        x = min(x * self.scale * sz, self.width - sz)
        tile = self.img.read_region((x, y), self.level, (sz, sz))
        tile = np.asarray(tile).reshape(sz, sz, 4)  # RGBA
        return tile[:, :, :3]  # RGB


class BFImageReader:
    def __init__(self, path, channel='DAPI'):
        import javabridge
        import bioformats as bf

        javabridge.start_vm(class_path=bf.JARS, run_headless=True)

        # It's a kind of magic lines that disable DEBUG logging
        JAVABRIDGE_DEFAULT_LOG_LEVEL = 'ERROR'
        rootLoggerName = javabridge.get_static_field("org/slf4j/Logger", "ROOT_LOGGER_NAME", "Ljava/lang/String;")
        rootLogger = javabridge.static_call("org/slf4j/LoggerFactory", "getLogger",
                                            "(Ljava/lang/String;)Lorg/slf4j/Logger;",
                                            rootLoggerName)
        jvm_log_level = javabridge.get_static_field("ch/qos/logback/classic/Level", JAVABRIDGE_DEFAULT_LOG_LEVEL,
                                                    "Lch/qos/logback/classic/Level;")
        javabridge.call(rootLogger, "setLevel", "(Lch/qos/logback/classic/Level;)V", jvm_log_level)

        # Retrieve image sizes
        meta = bf.get_omexml_metadata(path)
        meta = bf.OMEXML(meta).image().Pixels

        self.width = meta.get_SizeX()
        self.height = meta.get_SizeY()

        # Find an index of channel
        self.channel_id = None
        for i in range(meta.channel_count):
            if meta.Channel(i).Name == channel:
                self.channel_id = i
                break
        if self.channel_id is None:
            raise Exception('Unable to find channel ' + channel)

        self.reader = bf.ImageReader(path)


    def get_tile(self, tile_id, sz):
        x, y = tile_id
        y = min(y * sz, self.height - sz)  # Last tile offset
        x = min(x * sz, self.width - sz)
        data = self.reader.read(XYWH=((x, y, sz, sz)), rescale=False)
        return data[:, :, self.channel_id:self.channel_id + 1]


class Contour:
    def __init__(self, pts):
        self.pts = pts
        self.xmin, self.ymin = np.amin(pts, axis=0)
        self.xmax, self.ymax = np.amax(pts, axis=0)
        mask = np.zeros((self.ymax - self.ymin + 1,
                         self.xmax - self.xmin + 1), dtype=np.uint8)
        self.mask = cv.drawContours(mask, [pts - [self.xmin, self.ymin]], -1, (1), cv.FILLED)


def normalize_percentile(inp, percentiles=[1.0, 99.0]):
    num_channels = inp.shape[2]
    total = inp.shape[0] * inp.shape[1]
    num_colors = 256
    scale = []
    offset = []
    for ch in range(num_channels):
        # Compute a histogram for a channel
        hist_item = cv.calcHist([inp], [ch], None, [num_colors], [0, num_colors])

        # Find two percentiles from colors distribution
        counter = 0
        i = 0
        ind = int(percentiles[i] / 100.0 * total)
        rng = [0, 0]
        for color in range(num_colors):
            counter += hist_item[color]
            if counter >= ind:
                rng[i] = color
                if i == 1:
                    break
                else:
                    i += 1
                ind = int(percentiles[i] / 100.0 * total)
        scale.append(1.0 / (rng[1] - rng[0]))
        offset.append(-rng[0])

    return (inp + offset) * scale


def nms(coords, probs, nms_threshold, tile_size):
    # To optimize NMS we compare contours not with each other but with a global
    # predictions mask
    global_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)

    coords = [np.clip(pts, 0, tile_size - 1) for pts in coords]
    contours = [Contour(pts) for pts in coords]
    ids = []
    for i in reversed(np.argsort(probs)):
        ctr = contours[i]
        ref_mask = global_mask[ctr.ymin : ctr.ymax + 1, ctr.xmin : ctr.xmax + 1]

        inter_area = np.count_nonzero(np.logical_and(ref_mask, ctr.mask))
        union_area = np.count_nonzero(np.logical_or(ref_mask, ctr.mask))
        iou = inter_area / union_area
        if iou <= nms_threshold:
            ids.append(i)
            ref_mask[:, :] |= ctr.mask[:, :]

    return ids


class StarDistModel:
    def __init__(self, ie, args):
        self.net = ie.read_network(args.model)
        self.inp_name = next(iter(self.net.input_info.keys()))
        self.out_name = next(iter(self.net.outputs.keys()))

        angles = np.arange(0, 2 * pi, pi / 16)  # Get 32 angles
        self.cos_angles = np.cos(angles).reshape(-1, 1)
        self.sin_angles = np.sin(angles).reshape(-1, 1)
        self.nms_threshold = args.nms_threshold
        self.confidence_threshold = args.confidence_threshold


    def preprocess(self, tile):
        tile = normalize_percentile(tile)
        # NHWC to NCHW
        tile = np.expand_dims(tile.transpose(2, 0, 1), axis=0)
        return {self.inp_name: tile}, None


    def postprocess(self, outputs, meta):
        out = outputs[self.out_name]

        # Network predicts two concatenated tensors - probabilities of the nucleos
        # and the distances among an every anchor ray.
        probs = out[0, 0]
        distances = out[0, 1:]

        # Filter nucleor by probabilities.
        mask = probs > self.confidence_threshold
        probs = probs[mask]
        distances = distances[:, mask]
        centers = np.where(mask)

        # Compute the coordinates of contours
        coords = np.array([centers[1] + distances * self.cos_angles,
                           centers[0] + distances * self.sin_angles])
        coords = coords.reshape(2, 32, -1).transpose(2, 1, 0).astype(np.int32)
        ids = nms(coords, probs, self.nms_threshold, tile_size=out.shape[-1])
        return coords[ids]


def get_plugin_configs(device, num_streams, num_threads):
    config_user_specified = {}

    devices_nstreams = {}
    if num_streams:
        devices_nstreams = {device: num_streams for device in ['CPU', 'GPU'] if device in device} \
            if num_streams.isdigit() \
            else dict(device.split(':', 1) for device in num_streams.split(','))

    if 'CPU' in device:
        if num_threads is not None:
            config_user_specified['CPU_THREADS_NUM'] = str(num_threads)
        if 'CPU' in devices_nstreams:
            config_user_specified['CPU_THROUGHPUT_STREAMS'] = devices_nstreams['CPU'] \
                if int(devices_nstreams['CPU']) > 0 \
                else 'CPU_THROUGHPUT_AUTO'

    if 'GPU' in device:
        if 'GPU' in devices_nstreams:
            config_user_specified['GPU_THROUGHPUT_STREAMS'] = devices_nstreams['GPU'] \
                if int(devices_nstreams['GPU']) > 0 \
                else 'GPU_THROUGHPUT_AUTO'

    return config_user_specified


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Path to input image')
    parser.add_argument('-m', '--model', required=True,
                        help='Path to *.xml file of the model')
    parser.add_argument('-d', '--device', default='CPU',
                        help='Device to be utilized for deep learning inference')
    parser.add_argument('-t', '--confidence_threshold', default=0.7, type=float,
                        help='Probability threshold for detection')
    parser.add_argument('--nms', dest='nms_threshold', default=0.5, type=float,
                        help='NMS procedure threshold for IoU')
    parser.add_argument('--tile_size', default=1024, type=int, help='Tile size')
    parser.add_argument('--no_show', help="Optional. Don't show output.", action='store_true')

    infer_args = parser.add_argument_group('Inference options')
    infer_args.add_argument('-nireq', '--num_infer_requests', help='Optional. Number of infer requests',
                            default=0, type=int)
    infer_args.add_argument('-nstreams', '--num_streams',
                            help='Optional. Number of streams to use for inference on the CPU or/and GPU in throughput '
                                 'mode (for HETERO and MULTI device cases use format '
                                 '<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>).',
                            default='0', type=str)
    infer_args.add_argument('-nthreads', '--num_threads', default=None, type=int,
                            help='Optional. Number of threads to use for inference on CPU (including HETERO cases).')
    args = parser.parse_args()

    if args.input.endswith('.ndpi'):
        img = OpenSlideImageReader(args.input)
    else:
        img = BFImageReader(args.input)

    num_tiles_x = ceil(img.height / args.tile_size)
    num_tiles_y = ceil(img.width / args.tile_size)
    print('Number of tiles:', num_tiles_x * num_tiles_y)

    log.info('Initializing Inference Engine...')
    ie = IECore()

    plugin_config = get_plugin_configs(args.device, args.num_streams, args.num_threads)

    log.info('Loading network...')

    model = StarDistModel(ie, args)

    detector_pipeline = AsyncPipeline(ie, model, plugin_config,
                                      device=args.device, max_num_requests=args.num_infer_requests)

    tile_id = (0, 0)
    ready_tile_id = (0, 0)
    num_tiles_x = ceil(img.height / args.tile_size)
    num_tiles_y = ceil(img.width / args.tile_size)
    log.info('Number of tiles: {}'.format(num_tiles_x * num_tiles_y))

    def next_tile(tile_id):
        x, y = tile_id
        if x < num_tiles_x - 1:
            return (x + 1, y)
        elif y < num_tiles_y - 1:
            return (0, y + 1)
        else:
            return None

    start = time.time()

    while True:
        if detector_pipeline.callback_exceptions:
            raise detector_pipeline.callback_exceptions[0]
        # Process all completed requests
        results = detector_pipeline.get_result(ready_tile_id)
        if results:
            contours, frame_meta = results
            tile = frame_meta['tile']
            if not args.no_show:
                if tile.shape[2] == 1:
                    tile = np.repeat(tile, 3, axis=-1)
                else:
                    tile = cv.cvtColor(tile, cv.COLOR_RGB2BGR)

                # Draw detections
                tile = cv.drawContours(tile, contours, -1, (0, 0, 255))

                cv.imshow('StarDist with OpenVINO', tile)
                print(ready_tile_id)
                key = cv.waitKey()

                ESC_KEY = 27
                # Quit.
                if key in {ord('q'), ord('Q'), ESC_KEY}:
                    break

            ready_tile_id = next_tile(ready_tile_id)
            if ready_tile_id is None:
                break

        if detector_pipeline.is_ready():
            # Get new tile
            start_time = perf_counter()
            tile = img.get_tile(tile_id, args.tile_size)
            detector_pipeline.submit_data(tile, tile_id,
                                          meta={'tile': tile, 'start_time': start_time})
            tile_id = next_tile(tile_id)
            if tile_id is None:
                break
        else:
            # Wait for empty request
            detector_pipeline.await_any()

    log.info('Finished in {:.2f} seconds'.format(time.time() - start))


if __name__ == '__main__':
    sys.exit(main() or 0)
