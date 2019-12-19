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

import argparse
import time
import queue
from threading import Thread
import json
import logging as log

import cv2 as cv

from utils.network_wrappers import Detector, VectorCNN, MaskRCNN, \
                                   DetectionsFromFileReader, ReIDWithOrientationWrapper
from mc_tracker.mct import MultiCameraTracker
from utils.analyzer import save_embeddings
from utils.misc import read_py_config, check_pressed_keys, AverageEstimator, set_log_config
from utils.video import MulticamCapture, NormalizerCLAHE
from utils.visualization import visualize_multicam_detections
from openvino.inference_engine import IECore # pylint: disable=import-error,E0611

set_log_config()


class FramesThreadBody:
    def __init__(self, capture, max_queue_length=2):
        self.process = True
        self.frames_queue = queue.Queue()
        self.capture = capture
        self.max_queue_length = max_queue_length

    def __call__(self):
        while self.process:
            if self.frames_queue.qsize() > self.max_queue_length:
                time.sleep(0.1)
            has_frames, frames = self.capture.get_frames()
            if not has_frames and self.frames_queue.empty():
                self.process = False
                break
            if has_frames:
                self.frames_queue.put(frames)


def run(params, capture, detector, reid):
    win_name = 'Multi camera tracking'
    frame_number = 0
    avg_latency = AverageEstimator()
    key = -1
    config = {}
    if len(params.config):
        config = read_py_config(params.config)

    if config['normalizer_config']['enabled']:
        del config['normalizer_config']['enabled']
        capture.add_transform(NormalizerCLAHE(**config['normalizer_config']))

    tracker = MultiCameraTracker(capture.get_num_sources(), reid, config['sct_config'], **config['mct_config'],
                                 visual_analyze=config['analyzer'])

    thread_body = FramesThreadBody(capture, max_queue_length=len(capture.captures) * 2)
    frames_thread = Thread(target=thread_body)
    frames_thread.start()

    if len(params.output_video):
        video_output_size = (1920 // capture.get_num_sources(), 1080)
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        output_video = cv.VideoWriter(params.output_video,
                                      fourcc, 24.0,
                                      video_output_size)
    else:
        output_video = None

    prev_frames = thread_body.frames_queue.get()
    detector.run_asynch(prev_frames)

    while thread_body.process:
        key = check_pressed_keys(key)
        if key == 27:
            break
        start = time.time()
        try:
            frames = thread_body.frames_queue.get_nowait()
        except queue.Empty:
            frames = None

        if frames is None:
            continue

        if params.detections:
            all_detections = detector.get_detections(frame_number)
        else:
            all_detections = detector.wait_and_grab()
            detector.run_asynch(frames)

        all_masks = [[] for _ in range(len(all_detections))]
        for i, detections in enumerate(all_detections):
            all_detections[i] = [det[0] for det in detections]
            all_masks[i] = [det[2] for det in detections if len(det) == 3]

        tracker.process(prev_frames, all_detections, all_masks)
        tracked_objects = tracker.get_tracked_objects()

        latency = time.time() - start
        avg_latency.update(latency)
        fps = round(1. / latency, 1)

        vis = visualize_multicam_detections(prev_frames, tracked_objects, fps, **config['visualization_config'])
        cv.imshow(win_name, vis)
        if output_video:
            output_video.write(cv.resize(vis, video_output_size))

        print('\rProcessing frame: {}, fps = {} (avg_fps = {:.3})'.format(
                            frame_number, fps, 1. / avg_latency.get()), end="")
        frame_number += 1
        prev_frames, frames = frames, prev_frames
    print('')

    thread_body.process = False
    frames_thread.join()

    if len(params.history_file):
        history = tracker.get_all_tracks_history()
        with open(params.history_file, 'w') as outfile:
            json.dump(history, outfile)

    if len(config['embeddings']['save_path']):
        save_embeddings(tracker.scts, **config['embeddings'])


def main():
    """Prepares data for the person recognition demo"""
    parser = argparse.ArgumentParser(description='Multi camera multi person \
                                                  tracking live demo script')
    parser.add_argument('-i', type=str, nargs='+', help='Input sources (indexes \
                        of cameras or paths to video files)', required=True)
    parser.add_argument('--detections', type=str, nargs='+',
                        help='Json files with detections')

    parser.add_argument('-m', '--m_detector', type=str, required=True,
                        help='Path to the person detection model')
    parser.add_argument('--t_detector', type=float, default=0.6,
                        help='Threshold for the person detection model')

    parser.add_argument('--m_segmentation', type=str, required=False,
                        help='Path to the instance segmentation model')
    parser.add_argument('--t_segmentation', type=float, default=0.6,
                        help='Threshold for person instance segmentation model')

    parser.add_argument('--m_orientation', type=str, required=False,
                        help='Path to the people orientation classification model')
    parser.add_argument('--t_orientation', type=float, default=0.7,
                        help='Confidence threshold for people orientation clissifier')

    parser.add_argument('--m_reid', type=str, required=True,
                        help='Path to the person reidentification model')

    parser.add_argument('--output_video', type=str, default='', required=False)
    parser.add_argument('--config', type=str, default='', required=False)
    parser.add_argument('--history_file', type=str, default='', required=False)

    parser.add_argument('-d', '--device', type=str, default='CPU')
    parser.add_argument('-l', '--cpu_extension',
                        help='MKLDNN (CPU)-targeted custom layers.Absolute \
                              path to a shared library with the kernels impl.',
                             type=str, default=None)
    parser.add_argument("--no_show", help="Optional. Don't show output", action='store_true')

    args = parser.parse_args()

    capture = MulticamCapture(args.i)

    log.info("Creating Inference Engine")
    ie = IECore()

    if args.detections:
        person_detector = DetectionsFromFileReader(args.detections, args.t_detector)
    else:
        if args.m_segmentation:
            person_detector = MaskRCNN(ie, args.m_segmentation, args.t_segmentation,
                                       args.device, args.cpu_extension,
                                       capture.get_num_sources())
        else:
            person_detector = Detector(ie, args.m_detector, args.t_detector,
                                       args.device, args.cpu_extension,
                                       capture.get_num_sources())

    orientation_classifier = VectorCNN(ie, args.po_model, args.device, args.cpu_extension) \
        if args.m_orientation else None

    if args.m_reid:
        person_recognizer = VectorCNN(ie, args.m_reid, args.device, args.cpu_extension)
        person_recognizer = ReIDWithOrientationWrapper(person_recognizer,
                                                       orientation_classifier, args.t_orientation)
    else:
        person_recognizer = None

    run(args, capture, person_detector, person_recognizer)
    log.info('Demo finished successfully')


if __name__ == '__main__':
    main()
