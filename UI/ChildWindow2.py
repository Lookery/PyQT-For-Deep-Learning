# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ChildWindow2.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form2(object):
    def setupUi(self, Form2):
        Form2.setObjectName("Form2")
        Form2.resize(405, 228)
        self.label_2 = QtWidgets.QLabel(Form2)
        self.label_2.setGeometry(QtCore.QRect(10, 100, 91, 18))
        self.label_2.setObjectName("label_2")
        self.label = QtWidgets.QLabel(Form2)
        self.label.setGeometry(QtCore.QRect(10, 10, 101, 18))
        self.label.setObjectName("label")
        self.pushButton_2 = QtWidgets.QPushButton(Form2)
        self.pushButton_2.setGeometry(QtCore.QRect(10, 60, 80, 26))
        self.pushButton_2.setObjectName("pushButton_2")
        self.lineEdit_2 = QtWidgets.QLineEdit(Form2)
        self.lineEdit_2.setGeometry(QtCore.QRect(10, 120, 371, 26))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.pushButton = QtWidgets.QPushButton(Form2)
        self.pushButton.setGeometry(QtCore.QRect(300, 190, 80, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_3 = QtWidgets.QPushButton(Form2)
        self.pushButton_3.setGeometry(QtCore.QRect(10, 150, 80, 26))
        self.pushButton_3.setObjectName("pushButton_3")
        self.lineEdit = QtWidgets.QLineEdit(Form2)
        self.lineEdit.setGeometry(QtCore.QRect(10, 30, 371, 26))
        self.lineEdit.setObjectName("lineEdit")

        self.retranslateUi(Form2)
        QtCore.QMetaObject.connectSlotsByName(Form2)

    def retranslateUi(self, Form2):
        _translate = QtCore.QCoreApplication.translate
        Form2.setWindowTitle(_translate("Form2", "Form"))
        self.label_2.setText(_translate("Form2", "Test Path"))
        self.label.setText(_translate("Form2", "Model Path"))
        self.pushButton_2.setText(_translate("Form2", "Select"))
        self.pushButton.setText(_translate("Form2", "RUN"))
        self.pushButton_3.setText(_translate("Form2", "Select"))

