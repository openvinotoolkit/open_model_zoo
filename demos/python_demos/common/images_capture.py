import os
import copy

import cv2


class InvalidInput(Exception):
    message = ''

    def __init__(self, message):
        InvalidInput.message = message

    def __str__(self):
        return 'InvalidInput'


class OpenError(Exception):
    message = ''

    def __init__(self, message):
        OpenError.message = message

    def __str__(self):
        return 'OpenError'


class ImagesCapture:

    def __init__(self, loop):
        self.loop = loop

    def get_type(self):
        return self.type

    def read():
        raise NotImplementedError

    def get_fps(self):
        return 1

    def get_resolution():
        raise NotImplementedError


class ImreadWrapper(ImagesCapture):

    def __init__(self, input, loop):
        super().__init__(loop)
        if not os.path.isfile(input):
            raise InvalidInput("Can't find the image by {}".format(input))
        self.image = cv2.imread(input, cv2.IMREAD_COLOR)
        if self.image is None:
            raise OpenError("Can't open the image from {}".format(input))
        self.type = 'IMAGE'
        self.can_read = True

    def read(self):
        if self.loop:
            return copy.deepcopy(self.image)
        if self.can_read:
            self.can_read = False
            return copy.deepcopy(self.image)
        return None

    def get_resolution(self):
        height, width, _ = self.image.shape
        return (width, height)


class DirReader(ImagesCapture):

    def __init__(self, input, loop):
        super().__init__(loop)
        self.dir = input
        if not os.path.isdir(self.dir):
            raise InvalidInput("Can't find the dir by {}".format(input))
        self.names = sorted(os.listdir(self.dir))
        if not self.names:
            raise OpenError("The dir {} is empty".format(input))
        self.file_id = 0
        self.type = 'DIR'
        for name in self.names:
            filename = os.path.join(self.dir, name)
            image = cv2.imread(filename, cv2.IMREAD_COLOR)
            if image is not None:
                return
        raise OpenError("Can't read the first image from {}".format(input))

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

    def get_resolution(self):
        for name in self.names:
            filename = os.path.join(self.dir, name)
            image = cv2.imread(filename, cv2.IMREAD_COLOR)
            if image is not None:
                height, width, _ = image.shape
                return (width, height)


class VideoCapWrapper(ImagesCapture):

    def __init__(self, input, loop, camera_resolution):
        super().__init__(loop)
        self.cap = cv2.VideoCapture()
        try:
            status = self.cap.open(int(input))
            self.type = 'CAMERA'
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_resolution[1])
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except ValueError:
            if not os.path.isfile(input):
                raise InvalidInput("Can't find the video by {}".format(input))
            status = self.cap.open(input)
            self.type = 'VIDEO'
        if not status:
            raise OpenError("Can't open the {} from {}".format(self.get_type().lower(), input))

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

    def get_fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_resolution(self):
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))


def open_images_capture(input, loop, camera_resolution=(1280, 720)):
    errors = {'InvalidInput': [], 'OpenError': []}
    for idx, reader in enumerate((DirReader, ImreadWrapper, VideoCapWrapper)):
        try:
            if idx != 2:
                return reader(input, loop)
            return reader(input, loop, camera_resolution)
        except (InvalidInput, OpenError) as e:
            errors[str(e)].append(e.message)
    if not errors['OpenError']:
        print(*errors['InvalidInput'], sep='\n')
    else:
        print(*errors['OpenError'], sep='\n')
    exit(1)
