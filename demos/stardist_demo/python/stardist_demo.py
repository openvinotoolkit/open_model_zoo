import argparse
import numpy as np
import cv2 as cv
from math import pi, ceil
from openslide import OpenSlide
from openvino.inference_engine import IECore

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
        rng = get_percentiles_range(inp[:,:,i], percentiles)
        scale = 1.0 / (rng[1] - rng[0])
        offset = -rng[0]
        inp[:,:,i] = (inp[:,:,i] + offset) * scale
    return inp


def get_tile(img, x, y, tile_size):
    width = img.dimensions[0]
    height = img.dimensions[1]
    y = min(y * tile_size, height - tile_size)  # Last tile offset
    x = min(x * tile_size, width - tile_size)
    tile = img.read_region((x, y), 0, (tile_size, tile_size))
    return np.array(tile.getdata()).reshape(tile_size, tile_size, 4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input', required=True,
                        help='Path to *.ndpi image. In example, from http://openslide.cs.cmu.edu/download/openslide-testdata/Hamamatsu/')
    parser.add_argument('-m', dest='model', required=True,
                        help='Path to *.xml file of the model')
    parser.add_argument('-d', dest='device', default='CPU',
                        help='Device to be utilized for deep learning inference')
    parser.add_argument('--tile_size', default=1024, help='Tile size')
    args = parser.parse_args()

    sz = args.tile_size
    angles = np.arange(0, 2 * pi, pi / 16)  # Get 32 angles
    coords = np.array([np.cos(angles), np.sin(angles)])

    img = OpenSlide(args.input)
    width = img.dimensions[0]
    height = img.dimensions[1]
    num_tiles_x = ceil(height / sz)
    num_tiles_y = ceil(width / sz)
    print('Number of tiles:', num_tiles_x * num_tiles_y)

    ie = IECore()
    net = ie.read_network(args.model)
    exec_net = ie.load_network(net, args.device)

    predictions = [[None] * num_tiles_x for _ in range(num_tiles_y)]
    for y in range(num_tiles_y // 2 - 1, num_tiles_y // 2 + 1):
        for x in range(num_tiles_x // 2 - 1, num_tiles_x // 2 + 1):
            tile = get_tile(img, x, y, sz)

            # RGBA to RGB
            tile = tile[:,:,:3]
            tile = normalize_percentile(tile.astype(np.float32))

            # NHWC to NCHW
            tile = np.expand_dims(tile.transpose(2, 0, 1), axis=0)

            out = exec_net.infer({'input': tile})
            out = next(iter(out.values()))

            # Network predicts two concatenated tensors - probabilities of the nucleos
            # and the distances among an every anchor ray.
            probs = out[0, 0]
            distances = out[0, 1:]

            # Filter nucleor by probabilities.
            mask = probs > 0.8
            distances = distances[:, mask]
            centers = np.where(mask)

            # Compute the coordinates of contours
            coords = np.array([centers[1] + distances * np.cos(angles).reshape(-1, 1),
                            centers[0] + distances * np.sin(angles).reshape(-1, 1)])
            coords = coords.reshape(2, 32, -1).transpose(2, 1, 0).astype(np.int32)
            predictions[y][x] = coords


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
        preds = predictions[tile_y][tile_x]

        print('')
        print('Tile position:     ({},{})'.format(tile_x, tile_y))
        print('Number of objects: {}'.format(len(preds)))

        tile = get_tile(img, tile_x, tile_y, sz)

        # RGBA to BGR
        tile = tile[:,:,[2, 1, 0]].astype(np.uint8)
        tile = cv.drawContours(tile.copy(), preds, -1, (0, 0, 255))
        cv.imshow(WIN_NAME, tile)

        key = cv.waitKey()
        if key == KEY_ESC:
            break
        elif key == KEY_UP or key == KEY_DOWN:
            tile_y = max(0, min(tile_y + (1 if key == KEY_DOWN else -1), num_tiles_y - 1))
        elif key == KEY_LEFT or key == KEY_RIGHT:
            tile_x = max(0, min(tile_x + (1 if key == KEY_RIGHT else -1), num_tiles_x - 1))

