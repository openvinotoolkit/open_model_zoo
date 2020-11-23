import os
import copy

import cv2


class InvalidInput(Exception):
    pass


class ImagesCapture:

    def __init__(self, loop):
        self.loop = loop

    def read():
        raise NotImplementedError


class ImreadWrapper(ImagesCapture):

    def __init__(self, input, loop):
        super().__init__(loop)
        self.image = cv2.imread(input, cv2.IMREAD_COLOR)
        self.can_read = True
        if self.image is None:
            raise InvalidInput

    def read(self):
        if self.loop:
            return copy.deepcopy(self.image)
        if self.can_read:
            self.can_read = False
            return copy.deepcopy(self.image)
        return None


class DirReader(ImagesCapture):

    def __init__(self, input, loop):
        super().__init__(loop)
        self.dir = input
        if not os.path.isdir(self.dir):
            raise InvalidInput
        self.names = sorted(os.listdir(self.dir))
        if not self.names:
            raise InvalidInput
        self.file_id = 0
        for name in self.names:
            filename = os.path.join(self.dir, name)
            image = cv2.imread(filename, cv2.IMREAD_COLOR)
            if image is not None:
                return
        raise RuntimeError("Can't read the first image from {} dir".format(self.dir))

    def read(self):
        while self.file_id < len(self.names):
            filename = os.path.join(self.dir, self.names[self.file_id])
            image = cv2.imread(filename, cv2.IMREAD_COLOR)
            self.file_id += 1
            if image is not None:
                return image
        if self.loop:
            self.file_id = 0
            while self.file_id < len(self.names):
                filename = os.path.join(self.dir, self.names[self.file_id])
                image = cv2.imread(filename, cv2.IMREAD_COLOR)
                self.file_id += 1
                if image is not None:
                    return image
        return None


class VideoCapWrapper(ImagesCapture):

    def __init__(self, input, loop, camera_resolution):
        super().__init__(loop)
        self.cap = cv2.VideoCapture()
        try:
            status = self.cap.open(int(input))
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_resolution[1])
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        except ValueError:
            status = self.cap.open(input)
        if not status:
            raise InvalidInput

    def read(self):
        status, image = self.cap.read()
        if not status:
            if not self.loop:
                return None
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            status, image = self.cap.read()
            if not status:
                return None
        return image


def open_images_capture(input, loop, camera_resolution=(1280, 720)):
    try:
        return ImreadWrapper(input, loop)
    except InvalidInput:
        pass
    try:
        return DirReader(input, loop)
    except InvalidInput:
        pass
    try:
        return VideoCapWrapper(input, loop, camera_resolution)
    except InvalidInput:
        pass
    raise RuntimeError("Can't read {}".format(input))
