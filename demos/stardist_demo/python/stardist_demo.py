import argparse
from threading import Thread
from queue import Queue
from math import pi, ceil
import time

import numpy as np
import cv2 as cv
from openslide import OpenSlide
from openvino.inference_engine import IECore, StatusCode

def get_percentiles_range(inp, percentiles):
    inp = inp.reshape(-1)
    n = inp.shape[0]
    result = [0] * len(percentiles)
    inp_sorted = np.sort(inp.reshape(-1))
    for i in range(len(percentiles)):
        idx = int(percentiles[i] / 100 * n)
        result[i] = inp_sorted[idx]
    return result


def normalize_percentile(inp, percentiles=[1.0, 99.0]):
    for i in range(inp.shape[2]):
        rng = get_percentiles_range(inp[:, :, i], percentiles)
        scale = 1.0 / (rng[1] - rng[0])
        offset = -rng[0]
        inp[:, :, i] = (inp[:, :, i] + offset) * scale
    return inp


def get_tile(img, level, x, y, tile_size):
    width = img.dimensions[0]
    height = img.dimensions[1]
    scale = int(img.level_downsamples[level])
    y = min(y * scale * tile_size, height - tile_size)  # Last tile offset
    x = min(x * scale * tile_size, width - tile_size)
    tile = img.read_region((x, y), level, (tile_size, tile_size))
    return np.asarray(tile).reshape(tile_size, tile_size, 4)


def nms(contours, probs, nms_threshold):

    def get_iou(ctr1, ctr2, ctr1_area, ctr2_area, ctr1_mask, ctr2_mask):
        mask1, xmin1, ymin1, xmax1, ymax1 = ctr1_mask
        mask2, xmin2, ymin2, xmax2, ymax2 = ctr2_mask

        xmin = max(xmin1, xmin2)
        xmax = min(xmax1, xmax2)
        ymin = max(ymin1, ymin2)
        ymax = min(ymax1, ymax2)
        if xmax < xmin or ymax < ymin:
            return 0.0

        inter_mask = np.logical_and(mask1[ymin - ymin1 : ymax - ymin1,
                                          xmin - xmin1 : xmax - xmin1],
                                    mask2[ymin - ymin2 : ymax - ymin2,
                                          xmin - xmin2 : xmax - xmin2])
        inter_area = np.sum(inter_mask)
        union_area = ctr1_area + ctr2_area - inter_area
        return inter_area / union_area


    # Precompute all contours areas and masks once
    areas = [cv.contourArea(c) for c in contours]
    masks = []
    for ctr in contours:
        xmin, ymin = np.amin(ctr, axis=0)
        xmax, ymax = np.amax(ctr, axis=0)
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        mask = cv.drawContours(mask, [ctr - [xmin, ymin]], -1, (255), cv.FILLED)
        masks.append((mask, xmin, ymin, xmax, ymax))

        area = cv.contourArea(ctr)
        if area < 0:
            raise Exception('Negative area!')
        areas.append(area)

    ids = []
    for i in reversed(np.argsort(probs)):
        # Check this contour with contours with higher probabilities:
        keep = True
        for idx in ids:
            iou = get_iou(contours[i], contours[idx], areas[i], areas[idx], masks[i], masks[idx])
            if iou > nms_threshold:
                keep = False
                break
        if keep:
            ids.append(i)
    return ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input', required=True,
                        help='Path to *.ndpi image. In example, from http://openslide.cs.cmu.edu/download/openslide-testdata/Hamamatsu/')
    parser.add_argument('-m', dest='model', required=True,
                        help='Path to *.xml file of the model')
    parser.add_argument('-d', dest='device', default='CPU',
                        help='Device to be utilized for deep learning inference')
    parser.add_argument('-t', dest='confidence_threshold', default=0.7, type=float,
                        help='Probability threshold for detection')
    parser.add_argument('--nms', dest='nms_threshold', default=0.5, type=float,
                        help='NMS procedure threshold for IoU')
    parser.add_argument('--pixel_size', default=0.5, type=float,
                        help='Scale factor for image to perform detection')
    parser.add_argument('--tile_size', default=1024, help='Tile size')
    args = parser.parse_args()

    sz = args.tile_size
    angles = np.arange(0, 2 * pi, pi / 16)  # Get 32 angles
    coords = np.array([np.cos(angles), np.sin(angles)])

    img = OpenSlide(args.input)

    # Get level for a given scale
    level = None
    for i, scale in enumerate(img.level_downsamples):
        if args.pixel_size == 1.0 / scale:
            level = i
            break
    if not level:
        raise Exception('No level found for pixel size {}. Available levels are {}'.format(args.pixel_size, img.level_dimensions))

    width = img.level_dimensions[level][0]
    height = img.level_dimensions[level][1]
    num_tiles_x = ceil(height / sz)
    num_tiles_y = ceil(width / sz)
    print('Number of tiles:', num_tiles_x * num_tiles_y)

    #
    # Thread which performs tiling: data reading and preprocessing
    #
    tiles_queue = Queue()
    def tiling_thread_body():
        for y in range(num_tiles_y):
            for x in range(num_tiles_x):
                tile = get_tile(img, level, x, y, sz)

                # RGBA to RGB
                tile = tile[:,:,:3]
                tile = normalize_percentile(tile.astype(np.float32))

                # NHWC to NCHW
                tile = np.expand_dims(tile.transpose(2, 0, 1), axis=0)

                # Save raw tile data and position
                tiles_queue.put((tile, x, y))

    #
    # Thread which performs postprocessing
    #
    predictions_queue = Queue()
    detections = [[None] * num_tiles_x for _ in range(num_tiles_y)]

    def postprocess_thread_body():
        while True:
            out, tile_x, tile_y = predictions_queue.get()
            if out is None:  # Termination criteria
                break

            # Network predicts two concatenated tensors - probabilities of the nucleos
            # and the distances among an every anchor ray.
            probs = out[0, 0]
            distances = out[0, 1:]

            # Filter nucleor by probabilities.
            mask = probs > args.confidence_threshold
            probs = probs[mask]
            distances = distances[:, mask]
            centers = np.where(mask)

            # Compute the coordinates of contours
            coords = np.array([centers[1] + distances * np.cos(angles).reshape(-1, 1),
                               centers[0] + distances * np.sin(angles).reshape(-1, 1)])
            coords = coords.reshape(2, 32, -1).transpose(2, 1, 0).astype(np.int32)
            ids = nms(coords, probs, args.nms_threshold)
            detections[tile_y][tile_x] = coords[ids]


    tiling_thread = Thread(target=tiling_thread_body)
    tiling_thread.start()

    postprocess_thread = Thread(target=postprocess_thread_body)
    postprocess_thread.start()

    #
    # Inference loop
    #
    ie = IECore()
    net = ie.read_network(args.model)
    exec_net = ie.load_network(net, args.device, config={'CPU_THROUGHPUT_STREAMS': 'CPU_THROUGHPUT_AUTO'}, num_requests=0)
    inp_name = next(iter(exec_net.input_info.keys()))
    out_name = next(iter(exec_net.outputs.keys()))

    req_tile_pos = [None] * len(exec_net.requests)  # A list of tiles positions per request. None is initial value
    start_time = time.time()
    iter_id = 0
    while tiling_thread.is_alive() or not tiles_queue.empty():
        tile, x, y = tiles_queue.get()

        # Get idle infer request
        req_id = exec_net.get_idle_request_id()
        if req_id < 0:
            status = exec_net.wait(num_requests=1)
            if status != StatusCode.OK:
                raise Exception('Wait for idle request failed!')
            req_id = exec_net.get_idle_request_id()
            if req_id < 0:
                raise Exception('Invalid request id!')

        request = exec_net.requests[req_id]

        if req_tile_pos[req_id]:
            out_x, out_y = req_tile_pos[req_id]
            # Copy output prediction
            out = request.output_blobs[out_name].buffer
            predictions_queue.put((out.copy(), out_x, out_y))

            iter_id += 1
            if iter_id == 10:
                start_time = time.time()
            elif iter_id % 100 == 0:
                t = time.time() - start_time
                # Try to keep balance between tiles queue and predictions queue. They should be stable and not empty.
                # Too small tiles queue might give lower FPS because of not fully utilized device.
                # Too big predictions queue may lead to huge memory usage and lower FPS because of it
                # (increase confidence threshold to speed up postprocessing this way).
                print('Processed {} tiles. Tiles in queue: {}. Predictions queue: {}. Average FPS: {:.2f}'.format(
                        iter_id,
                        tiles_queue.qsize(),
                        predictions_queue.qsize(),
                        (iter_id - 10) / t))

        # Run inference
        req_tile_pos[req_id] = (x, y)
        request.async_infer({inp_name: tile})

    tiling_thread.join()  # Sanity join
    predictions_queue.put((None, None, None))
    postprocess_thread.join()
    print('Done!', time.time() - start_time)

    # Render single tile at the time. User might navigate between tiles by arrows.
    WIN_NAME = 'StarDist with OpenVINO'
    cv.namedWindow(WIN_NAME, cv.WINDOW_NORMAL)
    KEY_ESC = 27
    KEY_LEFT = 81
    KEY_UP = 82
    KEY_RIGHT = 83
    KEY_DOWN = 84
    tile_x = num_tiles_x // 2
    tile_y = num_tiles_y // 2
    while True:
        preds = detections[tile_y][tile_x]

        print('')
        print('Tile position:     ({},{})'.format(tile_x, tile_y))
        print('Number of objects: {}'.format(len(preds)))

        tile = get_tile(img, level, tile_x, tile_y, sz)

        # RGBA to BGR
        tile = tile[:, :, [2, 1, 0]].astype(np.uint8)
        tile = cv.drawContours(tile.copy(), preds, -1, (0, 0, 255))
        cv.imshow(WIN_NAME, tile)

        key = cv.waitKey()
        if key == KEY_ESC:
            break
        elif key == KEY_UP or key == KEY_DOWN:
            tile_y = max(0, min(tile_y + (1 if key == KEY_DOWN else -1), num_tiles_y - 1))
        elif key == KEY_LEFT or key == KEY_RIGHT:
            tile_x = max(0, min(tile_x + (1 if key == KEY_RIGHT else -1), num_tiles_x - 1))
