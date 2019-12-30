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
import sys
import cv2
import numpy as np
from argparse import ArgumentParser, SUPPRESS

from openvino.inference_engine import IECore

from detector import Detector

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("-i", "--input", type=str, nargs='+', default='', help="path to video or image/images")
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. The demo will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Path to labels mapping file", default=None, type=str)
    args.add_argument("-pt", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.3, type=float)
    args.add_argument("--no_show", help="Optional. Don't show output", action='store_true')

    return parser

class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx += 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        try:
            self.file_name = int(file_name[0])
        except:
            self.file_name = file_name[0]


    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img

def main():
    args = build_argparser().parse_args()

    ie = IECore()
    detector = Detector(ie, args.model, args.prob_threshold, args.device)

    img = cv2.imread(args.input[0], cv2.IMREAD_COLOR)
    frames_reader, delay = (VideoReader(args.input), 1) if img is None else (ImageReader(args.input), 0)

    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None

    for frame in frames_reader:
        detections = detector.detect(frame)
        for det in detections:
            xmin, ymin, xmax, ymax = det[:4].astype(np.int)
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(frame.shape[1], xmax)
            ymax = min(frame.shape[0], ymax)
            class_id = det[5]
            det_label = labels_map[int(class_id)] if labels_map else str(int(class_id))
            color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 3, 255))
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, det_label + ' ' + str(round(det[4] * 100, 1)) + ' %', (xmin, ymin - 7),
                         cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

        cv2.putText(frame, 'summary: {:.1f} FPS'.format(
            float(1 / (detector.infer_time * len(detections)))), (5, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 200))
        if args.no_show:
            continue
        cv2.imshow('CenterNet Detection Demo', frame)
        key = cv2.waitKey(delay)
        if key == 27:
            return

if __name__ == "__main__":
    sys.exit(main() or 0)

