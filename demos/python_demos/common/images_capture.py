import os
import copy

import cv2


class InvalidInput(Exception):
    pass


class ImagesCapture():

    def __init__(self, loop):
        self.loop = loop
        self.isVideo = False

    def read():
        pass


class ImageReader(ImagesCapture):

    def __init__(self, input, loop):
        super().__init__(loop)
        self.image = cv2.imread(input, cv2.IMREAD_COLOR)
        self.canRead = True
        if self.image is None:
            raise InvalidInput

    def read(self):
        if self.loop:
            return copy.deepcopy(self.image)
        if self.canRead:
            self.canRead = False
            return copy.deepcopy(self.image)
        return None


class DirReader(ImagesCapture):

    def __init__(self, input, loop):
        super().__init__(loop)
        self.dir = input
        if not os.path.isdir(self.dir):
            raise InvalidInput
        self.imageId = 0
        self.names = sorted(os.listdir(self.dir))
        if not self.names:
            raise InvalidInput
        for name in self.names:
            path = os.path.join(self.dir, name)
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                raise InvalidInput

    def read(self):
        while(True):
            if self.imageId < len(self.names): 
                path = os.path.join(self.dir, self.names[self.imageId])
                self.imageId += 1
                image = cv2.imread(path, cv2.IMREAD_COLOR)
                if image is None:
                    raise InvalidInput
                return copy.deepcopy(image)

            if self.loop:
                self.imageId = 0
                continue

            return None


class VideoReader(ImagesCapture):

    def __init__(self, input, loop, cameraResolution):
        super().__init__(loop)
        if not os.path.isfile(input):
            raise InvalidInput
        self.video = input
        self.isVideo = True
        self.cameraResolution = cameraResolution
        self.cap = cv2.VideoCapture(self.video)
        if not self.cap.isOpened():
            raise InvalidInput
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cameraResolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cameraResolution[1])

    def read(self):
        status, frame = self.cap.read()
        if not status:
            if not self.loop:
                return None
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            status, frame = self.cap.read()
            if not status:
                return None
        return frame


def openImagesCapture(input, loop, cameraResolution=(1280, 720)):
    input = input[0]
    try:
        return ImageReader(input, loop)
    except InvalidInput:
        pass
    try:
        return DirReader(input, loop)
    except InvalidInput: 
        pass
    try:
        return VideoReader(input, loop, cameraResolution)
    except InvalidInput:
        pass
    raise ValueError('Cannot read {}'.format(input))
