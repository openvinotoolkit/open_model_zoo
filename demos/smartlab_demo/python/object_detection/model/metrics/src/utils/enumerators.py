from enum import Enum


class MethodAveragePrecision(Enum):
    """
    Class representing if the coordinates are relative to the
    image size or are absolute values.

        Developed by: Rafael Padilla
        Last modification: Apr 28 2018
    """
    EVERY_POINT_INTERPOLATION = 1
    ELEVEN_POINT_INTERPOLATION = 2


class CoordinatesType(Enum):
    """
    Class representing if the coordinates are relative to the
    image size or are absolute values.

        Developed by: Rafael Padilla
        Last modification: Apr 28 2018
    """
    RELATIVE = 1
    ABSOLUTE = 2


class BBType(Enum):
    """
    Class representing if the bounding box is groundtruth or not.
    """
    GROUND_TRUTH = 1
    DETECTED = 2


class BBFormat(Enum):
    """
    Class representing the format of a bounding box.
    """
    XYWH = 1
    XYX2Y2 = 2
    PASCAL_XML = 3
    YOLO = 4


class FileFormat(Enum):
    ABSOLUTE_TEXT = 1
    PASCAL = 2
    LABEL_ME = 3
    COCO = 4
    CVAT = 5
    YOLO = 6
    OPENIMAGE = 7
    IMAGENET = 8
    UNKNOWN = 9
