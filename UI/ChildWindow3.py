# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ChildWindow3.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form3(object):
    def setupUi(self, Form3):
        Form3.setObjectName("Form3")
        Form3.resize(400, 300)
        self.lineEdit_2 = QtWidgets.QLineEdit(Form3)
        self.lineEdit_2.setGeometry(QtCore.QRect(10, 120, 361, 26))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.radioButton_3 = QtWidgets.QRadioButton(Form3)
        self.radioButton_3.setGeometry(QtCore.QRect(10, 230, 96, 24))
        self.radioButton_3.setObjectName("radioButton_3")
        self.lineEdit = QtWidgets.QLineEdit(Form3)
        self.lineEdit.setGeometry(QtCore.QRect(10, 30, 361, 26))
        self.lineEdit.setObjectName("lineEdit")
        self.label_4 = QtWidgets.QLabel(Form3)
        self.label_4.setGeometry(QtCore.QRect(10, 100, 91, 18))
        self.label_4.setObjectName("label_4")
        self.pushButton_3 = QtWidgets.QPushButton(Form3)
        self.pushButton_3.setGeometry(QtCore.QRect(10, 150, 80, 26))
        self.pushButton_3.setObjectName("pushButton_3")
        self.radioButton = QtWidgets.QRadioButton(Form3)
        self.radioButton.setGeometry(QtCore.QRect(240, 190, 151, 24))
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(Form3)
        self.radioButton_2.setGeometry(QtCore.QRect(10, 190, 96, 24))
        self.radioButton_2.setObjectName("radioButton_2")
        self.label_5 = QtWidgets.QLabel(Form3)
        self.label_5.setGeometry(QtCore.QRect(10, 10, 101, 18))
        self.label_5.setObjectName("label_5")
        self.pushButton_2 = QtWidgets.QPushButton(Form3)
        self.pushButton_2.setGeometry(QtCore.QRect(10, 60, 80, 26))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton = QtWidgets.QPushButton(Form3)
        self.pushButton.setGeometry(QtCore.QRect(290, 260, 80, 31))
        self.pushButton.setObjectName("pushButton")

        self.retranslateUi(Form3)
        QtCore.QMetaObject.connectSlotsByName(Form3)

    def retranslateUi(self, Form3):
        _translate = QtCore.QCoreApplication.translate
        Form3.setWindowTitle(_translate("Form3", "Form"))
        self.radioButton_3.setText(_translate("Form3", "SNV"))
        self.label_4.setText(_translate("Form3", "Save Path"))
        self.pushButton_3.setText(_translate("Form3", "Select"))
        self.radioButton.setText(_translate("Form3", "BaselineRemoval"))
        self.radioButton_2.setText(_translate("Form3", "Filter"))
        self.label_5.setText(_translate("Form3", "Read Path"))
        self.pushButton_2.setText(_translate("Form3", "Select"))
        self.pushButton.setText(_translate("Form3", "RUN"))

