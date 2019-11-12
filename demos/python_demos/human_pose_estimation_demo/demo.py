import argparse
import cv2
from detector import Detector
from estimator import HumanPoseEstimator

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-od-xml", type=str, required=True,
                        help="path to model of object detector in xml format")
    parser.add_argument("--model-od-bin", type=str, required=True,
                        help="path to model of object detector in bin format")
    parser.add_argument("--model-hpe-xml", type=str, required=True,
                        help="path to model of human pose estimator in xml format")
    parser.add_argument("--model-hpe-bin", type=str, required=True,
                        help="path to model of human pose estimator in bin format")
    parser.add_argument("--video", type=str, default='', help="path to video")
    parser.add_argument("--image", type=str, nargs='+',  default='', help="path to image or images")
    parser.add_argument("--device", type=str, default='CPU', required=False,
                        help="Specify the target to infer on CPU or GPU")
    parser.add_argument("--cpu-extension", type=str, required=False, help="path to cpu extension")
    parser.add_argument("--label-person", type=str, required=False, help="Label of class person for detector")

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
        self.file_name = file_name
        try:
            self.file_name = int(file_name)
        except ValueError:
            pass

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

    DetectorPerson = Detector(path_to_model_bin=args.model_od_bin,
                              path_to_model_xml=args.model_od_xml,
                              device=args.device,
                              path_to_lib=args.cpu_extension)

    SingleHumanPoseEstimator = HumanPoseEstimator(path_to_model_bin=args.model_hpe_bin,
                                                  path_to_model_xml=args.model_hpe_xml,
                                                  device=args.device,
                                                  path_to_lib=args.cpu_extension)

    if args.video == '' and args.image == '':
        raise ValueError('Either --video or --image has to be set')

    if args.video != '':
        frames_reader = VideoReader(args.video)
    else:
        frames_reader = ImageReader(args.image)

    for frame in frames_reader:
        bboxes = DetectorPerson.detect(frame)
        human_poses = [SingleHumanPoseEstimator.estimate(frame, bbox) for bbox in bboxes]

        colors = [(0, 0, 255),
                  (255, 0, 0), (0, 255, 0), (255, 0, 0), (0, 255, 0),
                  (255, 0, 0), (0, 255, 0), (255, 0, 0), (0, 255, 0),
                  (255, 0, 0), (0, 255, 0), (255, 0, 0), (0, 255, 0),
                  (255, 0, 0), (0, 255, 0), (255, 0, 0), (0, 255, 0)]

        for pose, bbox in zip(human_poses, bboxes):
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)
            for id_kpt, kpt in enumerate(pose):
                cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 3, colors[id_kpt], -1)

        cv2.putText(frame, 'summary fps: {:.2f} (fps estimation: {:.2f} / fps detection: {:.2f})'.format(
            float(1 / (DetectorPerson.infer_time + SingleHumanPoseEstimator.infer_time * len(human_poses))),
            float(1 / SingleHumanPoseEstimator.infer_time),
            float(1 / DetectorPerson.infer_time)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 200))

        cv2.imshow('frame', frame)
        key = cv2.waitKey(33)
        if key == 27:
            cv2.destroyAllWindows()
            return

if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_demo(args)
