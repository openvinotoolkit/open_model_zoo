# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'splash_ui.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(682, 206)
        Dialog.setMinimumSize(QtCore.QSize(682, 203))
        Dialog.setMaximumSize(QtCore.QSize(682, 206))
        Dialog.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.lbl_groundtruth_dir_2 = QtWidgets.QLabel(Dialog)
        self.lbl_groundtruth_dir_2.setGeometry(QtCore.QRect(10, 0, 681, 201))
        self.lbl_groundtruth_dir_2.setWordWrap(True)
        self.lbl_groundtruth_dir_2.setObjectName("lbl_groundtruth_dir_2")
        self.btn_Close = QtWidgets.QPushButton(Dialog)
        self.btn_Close.setGeometry(QtCore.QRect(530, 170, 141, 27))
        self.btn_Close.setObjectName("btn_Close")

        self.retranslateUi(Dialog)
        self.btn_Close.clicked.connect(Dialog.btn_close_clicked)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Information"))
        self.lbl_groundtruth_dir_2.setText(_translate("Dialog", "<html><head/><body><p>If you use this code for your research, please consider citing:<br/><br/><span style=\" font-weight:600;\">A Comparative Analysis of Object Detection Metrics with a Companion Open-Source Toolkit<br/><br/></span>Authors: Rafael Padilla, Wesley L. Passos, Thadeu L. B. Dias, Sergio L. Netto, Eduardo A. B. da Silva<br/>Journal: Electronics V. 10<br/>Year: 2021<br/>ISSN: 2079-9292<br/>DOI: 10.3390/electronics10030279<br/><br/><a href=\"https://github.com/rafaelpadilla/review_object_detection_metrics\"><span style=\" text-decoration: underline; color:#0000ff;\">https://github.com/rafaelpadilla/review_object_detection_metrics</span></a></p></body></html>"))
        self.btn_Close.setToolTip(_translate("Dialog", "The configurations will be applied in a random ground truth image."))
        self.btn_Close.setText(_translate("Dialog", "Close"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
