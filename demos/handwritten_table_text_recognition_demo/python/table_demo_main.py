import sys
import logging as log
import os
from argparse import ArgumentParser, SUPPRESS

import cv2
import numpy as np
from PySide2.QtWidgets import (QApplication, QMainWindow, QAction, QWidget,
                               QLabel, QLineEdit, QPushButton,
                               QVBoxLayout, QMessageBox, QFileDialog)
from PySide2.QtCore import Slot, Qt, QThread, Signal
from PySide2.QtGui import QImage, QPixmap

from ocr_worker import OcrWorker
from detection_worker import DetectionWorker
from utils.util import remove_redundant_rect, save_doc


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument("-m", "--model", type=str, required=True,
                      help="Required. Path to an .xml file with a trained model.")
    args.add_argument("-c", "--charlist", type=str, default=os.path.join(os.path.dirname(__file__), "data/scut_ept_char_list.txt"),
                      help="Optional. Path to the decoding char list file. Default is data/scut_ept_char_list.txt")
    args.add_argument("-d", "--device", type=str, default="CPU",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
                           "acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU")
    args.add_argument("-dc", "--designated_characters", type=str, default=None, help="Optional. Path to the designated character file")
    args.add_argument("-tk", "--top_k", type=int, default=20, help="Optional. Top k steps in looking up the decoded character, until a designated one is found")

    return parser


class Widget(QMainWindow):
    def __init__(self):
        super(Widget, self).__init__()
        self.args = build_argparser().parse_args()
        self.items = 0
        self.resize(1500, 750)
        self.detc_results = None
        self.reco_results = None
        self.lineEditContain = {}

        # Middle
        self.open = QPushButton('Open Image')
        self.detc = QPushButton('Detection')
        self.reco = QPushButton('Recognition')
        self.save = QPushButton('Save')
        self.quit = QPushButton('Exit')

        self.mid_widget = QWidget(self)
        self.mid_widget.setGeometry(700, 100, 150, 400)

        self.mid = QVBoxLayout(self.mid_widget)
        self.mid.setAlignment(Qt.AlignCenter)

        self.mid.addWidget(self.open)
        self.mid.addWidget(self.detc)
        self.mid.addWidget(self.reco)
        self.mid.addWidget(self.save)
        self.mid.addWidget(self.quit)

        self.detc.setEnabled(False)
        self.reco.setEnabled(False)
        self.save.setEnabled(False)

        # Left
        self.left_img_label = QLabel(self)
        self.left_img_label.setGeometry(0, 20, 700, 700)

        self.left_img_label.setText('left_img')
        self.left_img_label.setAlignment(Qt.AlignCenter)

        # Right
        self.right_widget = QWidget(self)
        self.right_widget.setGeometry(800, 20, 700, 700)
        self.right_img_label = QLabel(self.right_widget)
        self.right_img_label.setGeometry(0, 0, 700, 700)
        self.right_img_label.setText('right_img')
        self.right_img_label.setAlignment(Qt.AlignCenter)

        # layout
        self.statusBar().showMessage('Loading Model...')
        self._ocr_thread = OcrWorker(self.args)
        self.statusBar().showMessage('Model loaded，Please select the table image to be Detection')

        self.init_signals_slots()

    def init_signals_slots(self):
        # Signals and Slots
        self.detc.clicked.connect(self.detc_start)
        self.open.clicked.connect(self.open_file)
        self.quit.clicked.connect(self.quit_application)
        self.reco.clicked.connect(self.ocr_start)
        self.save.clicked.connect(self.save_file)

        self._ocr_thread.ocr_finished.connect(self.ocr_end)

    @Slot()
    def quit_application(self):
        QApplication.quit()

    @Slot()
    def clear_table(self):
        self.table.setRowCount(0)
        self.items = 0

    @Slot()
    def check_disable(self, s):
        if not self.description.text() or not self.price.text():
            self.add.setEnabled(False)
        else:
            self.add.setEnabled(True)

    @Slot()
    def detc_start(self):
        self._detection_thread.start()
        self.statusBar().showMessage('Table detection in progress...')
        self.detc.setEnabled(False)

    @Slot(np.ndarray, dict)
    def detc_end(self, img, detc_result):
        # update left image
        h, w, c = img.shape
        update_left_img = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
        self._pix = QPixmap.fromImage(update_left_img.scaledToHeight(700, Qt.SmoothTransformation))
        self.ratio = self._pix.height() / h
        self.left_img_label.setPixmap(self._pix)

        self.detc_results = remove_redundant_rect(detc_result)
        self._ocr_thread.get_detc_results(self.detc_results)
        self._ocr_thread.get_img(img)
        print('*' * 16)
        log.info('detc_result:')
        for it in self.detc_results:
            print(it, self.detc_results[it])

        print('*' * 16)
        self.reco.setEnabled(True)
        self.statusBar().showMessage('Table detection completed.')

    @Slot()
    def ocr_start(self):
        log.info('ocr start')
        self.statusBar().showMessage('Table text recognition in progress...')
        self._ocr_thread.start()
        self.reco.setEnabled(False)

    @Slot(dict)
    def ocr_end(self, results):
        self.reco_results = results
        self.right_img_label.setPixmap(self._pix)
        img_x = (700 - self._pix.width()) // 2
        log.info('Refresh GUI')
        # remove exist textlineEdit refresh ui
        for it in self.lineEditContain:
            self.lineEditContain[it].close()
        self.lineEditContain = {}

        detc_results = self.detc_results
        for it in detc_results:
            detc_results[it] = np.asarray(detc_results[it]) * self.ratio
            detc_results[it] = detc_results[it].astype(int)
            detc_results[it] = tuple(detc_results[it])
        print('*' * 16)
        print('ratio_detc_results:')
        for it in results:
            print(it, results[it])
        print('*' * 16)
        for it in detc_results:
            self.lineEditContain[it] = QLineEdit(self.right_widget)
            x, y, w, h = detc_results[it]
            self.lineEditContain[it].setGeometry(x + img_x, y, w, h)
            self.lineEditContain[it].setText(self.reco_results[it])
            self.lineEditContain[it].show()

        self.save.setEnabled(True)
        self.statusBar().showMessage('Table recognition has been completed, you can manually modify the error')

    @Slot()
    def unimplemented(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        msg.setText('unimplemented')
        msg.exec_()

    @Slot()
    def open_file(self):
        fname = QFileDialog.getOpenFileName(self, 'Open Image',
                                            os.path.curdir,
                                            'Image files (*.jpg *.png)')

        if fname[0] is not None:
            self._detection_thread = DetectionWorker(fname[0])
            self._detection_thread.detection_finished.connect(self.detc_end)

            img = cv2.imread(fname[0])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], img_rgb.shape[1] * 3,
                           QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image.scaledToHeight(700, Qt.SmoothTransformation))
            self.left_img_label.setPixmap(pixmap)
            self.detc.setEnabled(True)

    @Slot()
    def save_file(self):
        self.statusBar().showMessage('Please select a path to save')
        fname = QFileDialog.getSaveFileName(self, 'Save File',
                                            os.path.curdir,
                                            'docx files (*.docx)')

        if fname[0] is not None:
            path = fname[0]
            print('save to: ', path)
            reco_results = self.reco_results
            detc_results = self.detc_results
            container = self.lineEditContain
            for it in reco_results:
                reco_results[it] = container[it].text()
            save_doc(reco_results, detc_results, path)
            self.statusBar().showMessage('File saved to: {}'.format(path))


class MainWindow(QMainWindow):
    def __init__(self, widget):
        QMainWindow.__init__(self)
        self.setWindowTitle("OpenVINO Chinese Table Recognition Demo")

        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu("File")

        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.exit_app)

        self.file_menu.addAction(exit_action)
        self.setCentralWidget(widget)

    @Slot()
    def exit_app(self):
        QApplication.quit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = Widget()
    window = MainWindow(widget)
    window.resize(1600, 800)
    window.show()

    app.exec_()
