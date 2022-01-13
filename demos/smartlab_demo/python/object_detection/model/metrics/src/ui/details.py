import os
import random

import cv2
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QFileDialog, QMainWindow
from src.bounding_box import BoundingBox
from src.ui.details_ui import Ui_Dialog as Details_UI
from src.utils import general_utils
from src.utils.enumerators import BBType
from src.utils.general_utils import (add_bb_into_image, get_files_dir,
                                     remove_file_extension,
                                     show_image_in_qt_component)


class Details_Dialog(QMainWindow, Details_UI):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        # initialize variables
        self.dir_images = ''
        self.gt_annotations = None
        self.det_annotations = None
        self.text_statistics = '<b>#TYPE_BB#:</b><br>'
        self.text_statistics += '<br>* A total of <b>#TOTAL_BB#</b> bounding boxes were found in <b>#TOTAL_IMAGES#</b> images.'
        self.text_statistics += '<br>* The average area of the bounding boxes is <b>#AVERAGE_AREA_BB#</b> pixels.'
        self.text_statistics += '<br>* The amount of bounding boxes per class is:'
        self.text_statistics += '<br>#AMOUNT_BB_PER_CLASS#'
        self.lbl_sample_image.setScaledContents(True)
        # set maximum and minimum size
        self.setMaximumHeight(self.height())
        self.setMaximumWidth(self.width())
        # set selected image based on the list of images
        self.selected_image_index = 0

    def initialize_ui(self):
        # clear all information
        self.txb_statistics.setText('')
        self.lbl_sample_image.setText('')
        self.btn_previous_image.setEnabled(False)
        self.btn_next_image.setEnabled(False)
        # Create text with ground truth statistics
        if self.type_bb == BBType.GROUND_TRUTH:
            stats = self.text_statistics.replace('#TYPE_BB#', 'Ground Truth')
            self.annot_obj = self.gt_annotations
        elif self.type_bb == BBType.DETECTED:
            stats = self.text_statistics.replace('#TYPE_BB#', 'Detections')
            self.annot_obj = self.det_annotations
        self.chb_det_bb.setVisible(False)
        self.chb_gt_bb.setVisible(False)
        if self.det_annotations is not None and self.det_annotations != []:
            self.chb_det_bb.setVisible(True)
        if self.gt_annotations is not None and self.gt_annotations != []:
            self.chb_gt_bb.setVisible(True)
        stats = stats.replace('#TOTAL_BB#', str(len(self.annot_obj)))
        stats = stats.replace('#TOTAL_IMAGES#', str(BoundingBox.get_total_images(self.annot_obj)))
        stats = stats.replace('#AVERAGE_AREA_BB#',
                              '%.2f' % BoundingBox.get_average_area(self.annot_obj))
        # Get amount of bounding boxes per class
        self.bb_per_class = BoundingBox.get_amount_bounding_box_all_classes(self.annot_obj)
        amount_bb_per_class = 'No class found'
        if len(self.bb_per_class) > 0:
            amount_bb_per_class = ''
            longest_class_name = len(max(self.bb_per_class.keys(), key=len))
            for c, amount in self.bb_per_class.items():
                c = c.ljust(longest_class_name, ' ')
                amount_bb_per_class += f'   {c} : {amount}<br>'
        stats = stats.replace('#AMOUNT_BB_PER_CLASS#', amount_bb_per_class)
        self.txb_statistics.setText(stats)

        # get first image file and show it
        if os.path.isdir(self.dir_images):
            self.image_files = get_files_dir(
                self.dir_images, extensions=['jpg', 'jpge', 'png', 'bmp', 'tiff', 'tif'])
            if len(self.image_files) > 0:
                self.selected_image_index = 0
            else:
                self.selected_image_index = -1
        else:
            self.image_files = []
            self.selected_image_index = -1
        self.show_image()

    def show_image(self):
        if self.selected_image_index not in range(len(self.image_files)):
            self.btn_save_image.setEnabled(False)
            self.chb_gt_bb.setEnabled(False)
            self.chb_det_bb.setEnabled(False)
            self.lbl_sample_image.clear()
            self.lbl_image_file_name.setText('no image to show')
            return
        # Get all annotations and detections from this file
        if self.annot_obj is not None:
            # If Ground truth, bb will be drawn in green, red otherwise
            self.btn_previous_image.setEnabled(True)
            self.btn_next_image.setEnabled(True)
            self.btn_save_image.setEnabled(True)
            self.chb_gt_bb.setEnabled(True)
            self.chb_det_bb.setEnabled(True)
            self.lbl_image_file_name.setText(self.image_files[self.selected_image_index])
            # Draw bounding boxes
            self.loaded_image = self.draw_bounding_boxes()
            # Show image
            show_image_in_qt_component(self.loaded_image, self.lbl_sample_image)

    def draw_bounding_boxes(self):
        # Load image to obtain a clean image (without BBs)
        img_path = os.path.join(self.dir_images, self.image_files[self.selected_image_index])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Get bounding boxes of the loaded image
        img_name = self.image_files[self.selected_image_index]
        img_name = general_utils.get_file_name_only(img_name)
        # Add bounding boxes depending if the item is checked
        if self.chb_gt_bb.isChecked() and self.gt_annotations is not None:
            bboxes = BoundingBox.get_bounding_boxes_by_image_name(self.gt_annotations, img_name)
            if len(bboxes) == 0:
                bboxes = BoundingBox.get_bounding_boxes_by_image_name(self.gt_annotations, img_name)
            # Draw bounding boxes
            for bb in bboxes:
                img = add_bb_into_image(img, bb, color=(0, 255, 0), thickness=2, label=None)
        if self.chb_det_bb.isChecked() and self.det_annotations is not None:
            bboxes = BoundingBox.get_bounding_boxes_by_image_name(self.det_annotations, img_name)
            if len(bboxes) == 0:
                bboxes = BoundingBox.get_bounding_boxes_by_image_name(self.det_annotations,
                                                                      img_name)
            # Draw bounding boxes
            for bb in bboxes:
                img = add_bb_into_image(img, bb, color=(0, 0, 255), thickness=2, label=None)
        return img

    def show_dialog(self, type_bb, gt_annotations=None, det_annotations=None, dir_images=None):
        self.type_bb = type_bb
        self.gt_annotations = gt_annotations
        self.det_annotations = det_annotations
        self.dir_images = dir_images
        self.initialize_ui()
        self.show()

    def btn_plot_bb_per_classes_clicked(self):
        # dict_bbs_per_class = BoundingBox.get_amount_bounding_box_all_classes(gt_bbs, reverse=True)
        general_utils.plot_bb_per_classes(self.bb_per_class,
                                          horizontally=False,
                                          rotation=90,
                                          show=True)
        # plt.close()
        # plt.bar(self.bb_per_class.keys(), self.bb_per_class.values())
        # plt.xlabel('classes')
        # plt.ylabel('amount of bounding boxes')
        # plt.xticks(rotation=45)
        # plt.title('Bounding boxes per class')
        # fig = plt.gcf()
        # fig.canvas.set_window_title('Object Detection Metrics')
        # fig.show()

    # def btn_load_random_image_clicked(self):
    #     self.load_random_image()

    def btn_next_image_clicked(self):
        # If reached the last image, set index to start over
        if self.selected_image_index == len(self.image_files) - 1:
            self.selected_image_index = 0
        else:
            self.selected_image_index += 1
        self.show_image()

    def btn_previous_image_clicked(self):
        if self.selected_image_index == 0:
            self.selected_image_index = len(self.image_files) - 1
        else:
            self.selected_image_index -= 1
        self.show_image()

    def btn_save_image_clicked(self):
        dict_formats = {
            'PNG Image (*.png)': 'png',
            'JPEG Image (*.jpg, *.jpeg)': 'jpg',
            'TIFF Image (*.tif, *.tiff)': 'tif'
        }
        formats = ';;'.join(dict_formats.keys())
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, file_extension = QFileDialog.getSaveFileName(self,
                                                                "Save Image File",
                                                                "",
                                                                formats,
                                                                options=options)
        if file_name != '':
            # the extension was not informed, so add it
            if '.' not in file_name:
                file_name = file_name + '.' + dict_formats[file_extension]
            cv2.imwrite(file_name, cv2.cvtColor(self.loaded_image, cv2.COLOR_RGB2BGR))

    def chb_det_bb_clicked(self, state):
        # Draw bounding boxes
        self.loaded_image = self.draw_bounding_boxes()
        # Show image
        show_image_in_qt_component(self.loaded_image, self.lbl_sample_image)

    def chb_gt_bb_clicked(self, state):
        # Draw bounding boxes
        self.loaded_image = self.draw_bounding_boxes()
        # Show image
        show_image_in_qt_component(self.loaded_image, self.lbl_sample_image)
