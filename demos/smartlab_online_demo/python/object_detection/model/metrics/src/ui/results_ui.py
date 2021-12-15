# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'results_ui.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form_results(object):
    def setupUi(self, Form_results):
        Form_results.setObjectName("Form_results")
        Form_results.setWindowModality(QtCore.Qt.ApplicationModal)
        Form_results.resize(804, 500)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Form_results.sizePolicy().hasHeightForWidth())
        Form_results.setSizePolicy(sizePolicy)
        Form_results.setMinimumSize(QtCore.QSize(804, 500))
        Form_results.setMaximumSize(QtCore.QSize(804, 500))
        self.lbl_groundtruth_dir_23 = QtWidgets.QLabel(Form_results)
        self.lbl_groundtruth_dir_23.setGeometry(QtCore.QRect(10, 10, 781, 21))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.lbl_groundtruth_dir_23.setFont(font)
        self.lbl_groundtruth_dir_23.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_groundtruth_dir_23.setObjectName("lbl_groundtruth_dir_23")
        self.txb_results = QtWidgets.QTextBrowser(Form_results)
        self.txb_results.setGeometry(QtCore.QRect(10, 40, 781, 401))
        self.txb_results.setReadOnly(True)
        self.txb_results.setObjectName("txb_results")
        self.lbl_groundtruth_dir_4 = QtWidgets.QLabel(Form_results)
        self.lbl_groundtruth_dir_4.setGeometry(QtCore.QRect(10, 450, 781, 17))
        self.lbl_groundtruth_dir_4.setObjectName("lbl_groundtruth_dir_4")
        self.lbl_folder_output = QtWidgets.QLabel(Form_results)
        self.lbl_folder_output.setGeometry(QtCore.QRect(10, 475, 781, 17))
        self.lbl_folder_output.setObjectName("lbl_folder_output")

        self.retranslateUi(Form_results)
        QtCore.QMetaObject.connectSlotsByName(Form_results)

    def retranslateUi(self, Form_results):
        _translate = QtCore.QCoreApplication.translate
        Form_results.setWindowTitle(_translate("Form_results", "Results"))
        self.lbl_groundtruth_dir_23.setText(_translate("Form_results", "Results"))
        self.txb_results.setHtml(_translate("Form_results", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.lbl_groundtruth_dir_4.setText(_translate("Form_results", "Plots with Precision x Recall  curve per class calculated with PASCAL VOC were also saved in the folder: "))
        self.lbl_folder_output.setText(_translate("Form_results", "FOLDER"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form_results = QtWidgets.QWidget()
    ui = Ui_Form_results()
    ui.setupUi(Form_results)
    Form_results.show()
    sys.exit(app.exec_())

