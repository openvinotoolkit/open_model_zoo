import argparse
import cv2

from openvino.inference_engine import IECore

from detector import Detector
from estimator import HumanPoseEstimator

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m_od", "--model_od", type=str, required=True,
                        help="path to model of object detector in xml format")

    parser.add_argument("-m_hpe", "--model_hpe", type=str, required=True,
                        help="path to model of human pose estimator in xml format")

    parser.add_argument("-i", "--input", type=str, nargs='+', default='', help="path to video or image/images")
    parser.add_argument("-d", "--device", type=str, default='CPU', required=False,
                        help="Specify the target to infer on CPU or GPU")
    parser.add_argument("--person_label", type=int, required=False, default=15, help="Label of class person for detector")
    parser.add_argument("--no_show", help='Optional. Do not display output.', action='store_true')

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


def run_demo(args):
    ie = IECore()
    detector_person = Detector(ie, path_to_model_xml=args.model_od,
                              device=args.device,
                              label_class=args.person_label)

    single_human_pose_estimator = HumanPoseEstimator(ie, path_to_model_xml=args.model_hpe,
                                                  device=args.device)
    if args.input != '':
        img = cv2.imread(args.input[0], cv2.IMREAD_COLOR)
        frames_reader, delay = (VideoReader(args.input), 1) if img is None else (ImageReader(args.input), 0)
    else:
        raise ValueError('--input has to be set')

    for frame in frames_reader:
        bboxes = detector_person.detect(frame)
        human_poses = [single_human_pose_estimator.estimate(frame, bbox) for bbox in bboxes]

        colors = [(0, 0, 255),
                  (255, 0, 0), (0, 255, 0), (255, 0, 0), (0, 255, 0),
                  (255, 0, 0), (0, 255, 0), (255, 0, 0), (0, 255, 0),
                  (255, 0, 0), (0, 255, 0), (255, 0, 0), (0, 255, 0),
                  (255, 0, 0), (0, 255, 0), (255, 0, 0), (0, 255, 0)]

        for pose, bbox in zip(human_poses, bboxes):
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)
            for id_kpt, kpt in enumerate(pose):
                cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 3, colors[id_kpt], -1)

        cv2.putText(frame, 'summary: {:.1f} FPS (estimation: {:.1f} FPS / detection: {:.1f} FPS)'.format(
            float(1 / (detector_person.infer_time + single_human_pose_estimator.infer_time * len(human_poses))),
            float(1 / single_human_pose_estimator.infer_time),
            float(1 / detector_person.infer_time)), (5, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 200))
        if args.no_show:
            continue
        cv2.imshow('Human Pose Estimation Demo', frame)
        key = cv2.waitKey(delay)
        if key == 27:
            return

if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_demo(args)
