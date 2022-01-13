import os
import random

import cv2
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QFileDialog, QMainWindow
from src.bounding_box import BoundingBox
from src.ui.results_ui import Ui_Form_results as Results_UI


class Results_Dialog(QMainWindow, Results_UI):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)

    def show_dialog(self, coco_results, pascal_results, folder_results):
        tag_value = '<p style=" margin-top:0px; margin-bottom:0px;">VALUE</p>'
        text = ''
        if len(coco_results) != 0:
            text += '<span style=" font-weight:600;">COCO METRICS:</span>'
            for metric, res in coco_results.items():
                text += tag_value.replace('VALUE', f'{metric}: {res}')
        if len(pascal_results) != 0:
            for metric, res in pascal_results.items():
                if metric == 'per_class':
                    text += '<br />'
                    text += '<span style=" font-weight:600;">PASCAL METRIC (AP per class)</span>'
                    for c, ap in res.items():
                        text += tag_value.replace('VALUE', f'{c}: {ap["AP"]}')
                elif metric == 'mAP':
                    text += '<br />'
                    text += '<span style=" font-weight:600;">PASCAL METRIC (mAP)</span>'
                    text += tag_value.replace('VALUE', f'mAP: {res}')
        self.txb_results.setText(text)
        self.lbl_folder_output.setText(folder_results)
        self.show()
