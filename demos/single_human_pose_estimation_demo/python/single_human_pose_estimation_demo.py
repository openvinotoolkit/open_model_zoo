#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys
import logging as log
from time import perf_counter
import cv2

from openvino import Core, get_version

from detector import Detector
from estimator import HumanPoseEstimator

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python/model_zoo'))

import monitors
from images_capture import open_images_capture
from model_api.performance_metrics import PerformanceMetrics

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m_od", "--model_od", type=Path, required=True,
                        help="Required. Path to model of object detector in .xml format.")
    parser.add_argument("-m_hpe", "--model_hpe", type=Path, required=True,
                        help="Required. Path to model of human pose estimator in .xml format.")
    parser.add_argument("-i", "--input", required=True,
                        help="Required. An input to process. The input must be a single image, "
                             "a folder of images, video file or camera id.")
    parser.add_argument("--loop", default=False, action="store_true",
                        help="Optional. Enable reading the input in a loop.")
    parser.add_argument("-o", "--output", required=False,
                        help="Optional. Name of the output file(s) to save. Frames of odd width or height can be truncated. See https://github.com/opencv/opencv/pull/24086")
    parser.add_argument("-limit", "--output_limit", required=False, default=1000, type=int,
                      help="Optional. Number of frames to store in output. "
                           "If 0 is set, all frames are stored.")
    parser.add_argument("-d", "--device", type=str, default='CPU', required=False,
                        help="Optional. Specify the target to infer on CPU or GPU.")
    parser.add_argument("--person_label", type=int, required=False, default=15, help="Optional. Label of class person for detector.")
    parser.add_argument("--no_show", help='Optional. Do not display output.', action='store_true')
    parser.add_argument("-u", "--utilization_monitors", default="", type=str,
                        help="Optional. List of monitors to show initially.")
    return parser


def run_demo(args):
    cap = open_images_capture(args.input, args.loop)

    log.info('OpenVINO Runtime')
    log.info('\tbuild: {}'.format(get_version()))
    core = Core()

    log.info('Reading Object Detection model {}'.format(args.model_od))
    detector_person = Detector(core, args.model_od,
                               device=args.device,
                               label_class=args.person_label)
    log.info('The Object Detection model {} is loaded to {}'.format(args.model_od, args.device))

    log.info('Reading Human Pose Estimation model {}'.format(args.model_hpe))
    single_human_pose_estimator = HumanPoseEstimator(core, args.model_hpe,
                                                     device=args.device)
    log.info('The Human Pose Estimation model {} is loaded to {}'.format(args.model_hpe, args.device))

    delay = int(cap.get_type() in ('VIDEO', 'CAMERA'))
    video_writer = cv2.VideoWriter()

    frames_processed = 0
    presenter = monitors.Presenter(args.utilization_monitors, 25)
    metrics = PerformanceMetrics()

    start_time = perf_counter()
    frame = cap.read()
    if frame is None:
        raise RuntimeError("Can't read an image from the input")

    if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'),
                                             cap.fps(), (frame.shape[1], frame.shape[0])):
        raise RuntimeError("Can't open video writer")

    while frame is not None:
        bboxes = detector_person.detect(frame)
        human_poses = [single_human_pose_estimator.estimate(frame, bbox) for bbox in bboxes]

        presenter.drawGraphs(frame)

        colors = [(0, 0, 255),
                  (255, 0, 0), (0, 255, 0), (255, 0, 0), (0, 255, 0),
                  (255, 0, 0), (0, 255, 0), (255, 0, 0), (0, 255, 0),
                  (255, 0, 0), (0, 255, 0), (255, 0, 0), (0, 255, 0),
                  (255, 0, 0), (0, 255, 0), (255, 0, 0), (0, 255, 0)]

        for pose, bbox in zip(human_poses, bboxes):
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)
            for id_kpt, kpt in enumerate(pose):
                cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 3, colors[id_kpt], -1)

        metrics.update(start_time, frame)

        frames_processed += 1
        if video_writer.isOpened() and (args.output_limit <= 0 or frames_processed <= args.output_limit):
            video_writer.write(frame)

        if not args.no_show:
            cv2.imshow('Human Pose Estimation Demo', frame)
            key = cv2.waitKey(delay)
            if key == 27:
                break
            presenter.handleKey(key)

        start_time = perf_counter()
        frame = cap.read()

    metrics.log_total()
    for rep in presenter.reportMeans():
        log.info(rep)


if __name__ == "__main__":
    args = build_argparser().parse_args()
    sys.exit(run_demo(args) or 0)
