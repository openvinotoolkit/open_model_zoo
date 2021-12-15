from typing import Any

import numpy as np

from src.utils.general_utils import convert_box_xywh2xyxy


class Tube(object):
    """Tube object
    """
    def __init__(self, category_id: int, video_id: int, track: list, **kwargs: Any) -> None:
        """class constructor

        Args:
            category_id (int): class category of corresponding tube
            video_id (int): video id of corresponding tube
            track (list of dicts): list of dictionaries with keys "frame" and "bbox"
        """
        self.category_id = category_id
        self.video_id = video_id
        self.confidence = None

        self.track = {attr: np.array([det[attr] for det in track]) for attr in track[0]}

        if 'confidence' in self.track.keys():
            self.confidence = self.__compute_tube_confidence(self.track['confidence'])

        self.volume = self.__compute_tube_volume()

        self.__kwargs = kwargs
        for k, v in kwargs.items():
            super().__setattr__(k, v)

        # convert tube boxes from xywh to xyxy format
        boxes = convert_box_xywh2xyxy(self.get_boxes())
        self.track['bbox'] = boxes

    def __len__(self) -> int:
        return len(self.track)

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "video_id={}, ".format(self.video_id)
        s += "category_id={}, ".format(self.category_id)
        s += "tube_confidence={:.2f}, ".format(self.confidence)
        s += "{}, ".format("".join(f"{k}={v}" for k, v in self.__kwargs.items()))
        s += "track=[{}]".format(", ".join(f"{k}: {v}" for k, v in self.track.items()))
        s += ")"

        return s

    def __compute_tube_confidence(self, frames_confidences: list) -> float:
        """Compute tube confidence. It is the mean of individual bboxes along the frames.

        Args:
            frames_confidences (list): list containing the frames confidences

        Returns:
            numpy: the tube confidence
        """
        return np.mean(frames_confidences)

    def get_frames(self) -> list:
        """Return a list of frames of a given track.

        Returns:
            list: list of frames of tube.
        """
        return self.track['frame']

    def get_boxes(self) -> list:
        """Return a list of bboxes of a given track.

        Returns:
            list: list of bboxes of tube.
        """
        return self.track['bbox']

    def get_frame_boxes(self, frame_idx):
        """return the boxes of frame `frame_idx`

        Args:
            frame_idx (int): frame index

        Returns:
            list: list of bounding boxes of the `frame_idx` frame
        """
        frames = self.get_frames()
        indexes = [i for i, e in enumerate(frames) if e == frame_idx]

        boxes = self.get_boxes()[indexes]

        return boxes

    def __compute_tube_volume(self) -> float:
        """Compute tube volume. It is the summation of pixels of all bouding boxes that the tube contains.

        Returns:
            float: tube volume
        """
        boxes = self.get_boxes()
        areas = np.prod(boxes[:, 2:], axis=1)
        vol = areas.sum()
        return vol

    def get_tube_volume(self) -> float:
        """return the tube volume previously computed in self.__compute_tube_volume

        Returns:
            float: tube volume
        """
        return self.volume

    __repr__ = __str__
