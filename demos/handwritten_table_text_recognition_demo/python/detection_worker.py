from PySide2.QtCore import QThread, Signal
import numpy as np

from utils.perspective_transform import perspective_transform
from utils.table_detection import table_test


class DetectionWorker(QThread):
    detection_finished = Signal(np.ndarray, dict)

    def __init__(self, input_img):
        super().__init__()
        self.input_img = input_img

    def run(self):
        ptres = perspective_transform(self.input_img)
        detc_results, image_res = table_test(ptres)
        self.detection_finished.emit(image_res, detc_results)
