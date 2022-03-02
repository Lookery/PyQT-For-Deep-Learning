

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
from UI import *
from train import *

import datetime
import sys

sys.setrecursionlimit(15000)
log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
count = 1
global timer


class Stream(QObject):
    """Redirects console output to text widget."""
    newText = pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))
        QApplication.processEvents()

    def flush(self):
        pass


class MyTable(QMainWindow, Ui_MainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        loadUi("UI/mainWindow.ui", self)
        self.setWindowTitle("Exosome & Raman Example GUI")
        self.addToolBar(NavigationToolbar(self.mplwidget.canvas, self))
        self.addToolBar(NavigationToolbar(self.mplwidget2.canvas, self))
        # Custom output stream.

        sys.stdout = Stream(newText=self.on_update_text)
        self.Train_Start.clicked.connect(self.slot1)
        self.Train_Stop.clicked.connect(self.train_stop)
        self.pushButton_2.clicked.connect(self.slot2)
        self.Dataset_Start.clicked.connect(self.slot3)

        self.thread = NewThread()
        self.thread.sinOut.connect(self.thread_return)
        self.progressBar_Epoch.setVisible(False)
        self.progressBar_Epoch.setValue(0)
        self.Train_Stop.setVisible(False)
        self.ChildDialog1 = TrainWin()
        self.ChildDialog2 = TestWin()
        self.ChildDialog3 = PreprocessWin()
        self.epoch_num = 100

    def slot1(self):
        self.ChildDialog1.showNormal()
        self.ChildDialog1._signal.connect(self.model_train)

    def slot2(self):
        self.ChildDialog2.showNormal()
        self.ChildDialog2._signal.connect(self.model_test)

    def slot3(self):
        self.ChildDialog3.showNormal()
        self.ChildDialog3._signal.connect(self.data_process)

    def model_train(self, parameter, parameter1, parameter2, parameter3):
        # self.pushButton_1.setText("STOP")

        self.ChildDialog1.close()
        self.Train_Start.setVisible(False)
        self.Train_Stop.setVisible(True)
        self.progressBar_Epoch.setVisible(True)
        data_path = parameter
        model_path = parameter1
        self.epoch_num = parameter2
        learning_rate = parameter3
        self.thread.init(data_path, model_path, learning_rate, self.epoch_num)
        self.thread.start()

    def thread_return(self, epoch):
        step = (epoch + 1)*100/self.epoch_num
        self.progressBar_Epoch.setValue(step)
        if epoch == self.epoch_num:
            self.Train_Start.setVisible(True)
            self.Train_Stop.setVisible(False)
            self.progressBar_Epoch.setVisible(False)
        # if epoch >= 100:
        #     return 100
        # else:
        #     timer = threading.Timer(5, self.show_time)
        #     timer.start()

    def train_stop(self):
        self.thread.train.train_FLAG = False
        self.Train_Start.setVisible(True)
        self.Train_Stop.setVisible(False)
        self.progressBar_Epoch.setVisible(False)


    def model_test(self, parameter, parameter1):
        model_path = parameter
        test_path = parameter1
        self.ChildDialog2.showMinimized()
        res, x, plot, path = self.tensorflow_test(model_path, test_path)
        self.mplwidget.canvas.axes.clear()
        self.mplwidget.canvas.axes.plot(plot, x, color='red', linewidth=2.0)
        # self.mplwidget.canvas.axes.legend(["raman"], loc='upper right')
        self.mplwidget.canvas.axes.set_title("Raman")
        self.mplwidget.canvas.draw()
        self.lineEdit.setText(str(res))
        self.textEdit.setText(str(path))

    def data_process(self, parameter, parameter1):
        read_path = parameter
        save_path = parameter1
        self.ChildDialog3.showMinimized()
        self.preprocessor(read_path, save_path)

    def on_update_text(self, text):
        """Write console output to text widget."""
        cursor = self.process.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.process.setTextCursor(cursor)
        self.process.ensureCursorVisible()

    def closeEvent(self, event):
        """Shuts down application on close."""
        # Return stdout to defaults.
        sys.stdout = sys.__stdout__
        super().closeEvent(event)


class TrainWin(QMainWindow, Ui_Form):
    # 定义信号
    _signal = pyqtSignal(str, str, int, float)

    def __init__(self):
        super(TrainWin, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.setWindowTitle('Data Mode')
        self.pushButton.clicked.connect(self.slot1)
        self.pushButton_2.clicked.connect(self.slot2)
        self.pushButton_3.clicked.connect(self.slot3)
        self.lineEdit.setText('covid')
        self.lineEdit_2.setText('D:/DL/UiProject/saved_models/covid')
        self.doubleSpinBox.setValue(0.0001)
        self.spinBox.setValue(100)

    def slot1(self):
        data_str = self.lineEdit.text()
        data_str2 = self.lineEdit_2.text()
        data_str3 = self.spinBox.value()
        data_str4 = self.doubleSpinBox.value()
        # 发送信号
        self._signal.emit(data_str, data_str2, data_str3, data_str4)
        self.close()

    def slot2(self):
        filename = QFileDialog.getExistingDirectory(self, '选择文件夹', 'data')
        # QFileDialog.getOpenFileNames(None, "请选择要添加的文件", path, "Text Files (*.xls);;All Files (*)")
        self.lineEdit.setText(filename.split('/')[-1])
        # self.lineEdit.setReadOnly(True)

    def slot3(self):
        filename = QFileDialog.getExistingDirectory(self, '选择文件夹', 'data')
        self.lineEdit_2.setText(filename)
        self.lineEdit_2.setReadOnly(True)


class TestWin(QMainWindow, Ui_Form2):
    # 定义信号
    _signal = pyqtSignal(str, str)

    def __init__(self):
        super(TestWin, self).__init__()

        self.setupUi(self)
        self.retranslateUi(self)
        self.setWindowTitle('Path Select')

        self.pushButton.clicked.connect(self.slot1)

        self.pushButton_2.clicked.connect(self.slot2)

        self.pushButton_3.clicked.connect(self.slot3)

        self.lineEdit.setText('D:/DL/Project/saved_models/keras_%s_model.h5')
        self.lineEdit_2.setText('D:/DL/Project/Old Project/data/origin')

    def slot1(self):
        data_str = self.lineEdit.text()
        data_str2 = self.lineEdit_2.text()
        # 发送信号
        self._signal.emit(data_str, data_str2)

        self.close()

    def slot2(self):
        filename, filetype = QFileDialog.getOpenFileName(self, '选择文件', 'data', 'All files(*)')
        self.lineEdit.setText(filename)
        self.lineEdit.setReadOnly(True)

    def slot3(self):
        filename, filetype = QFileDialog.getOpenFileName(self, '选择文件', 'data', 'All files(*)')
        self.lineEdit_2.setText(filename)
        self.lineEdit_2.setReadOnly(True)


class PreprocessWin(QMainWindow, Ui_Form3):
    # 定义信号
    _signal = pyqtSignal(str, str)

    def __init__(self):
        super(PreprocessWin, self).__init__()

        self.setupUi(self)
        self.retranslateUi(self)
        self.setWindowTitle('Path Select')

        self.pushButton.clicked.connect(self.slot1)

        self.pushButton_2.clicked.connect(self.slot2)

        self.pushButton_3.clicked.connect(self.slot3)

        self.lineEdit.setText('D:/DL/Project/Old Project/data/origin')
        self.lineEdit_2.setText('D:/DL/Project/Old Project/data/renew')

    def slot1(self):
        data_str = self.lineEdit.text()
        data_str2 = self.lineEdit_2.text()
        # 发送信号
        self._signal.emit(data_str, data_str2)

        self.close()

    def slot2(self):
        filename = QFileDialog.getExistingDirectory(self, '选择文件夹', 'data')
        self.lineEdit.setText(filename)
        self.lineEdit.setReadOnly(True)

    def slot3(self):
        filename = QFileDialog.getExistingDirectory(self, '选择文件夹', 'data')
        self.lineEdit_2.setText(filename)
        self.lineEdit_2.setReadOnly(True)


class NewThread(QThread):
    sinOut = pyqtSignal(int)
    handle = -1

    # 创建了一个newThread的线程
    def __init__(self, parent=None):
        super(NewThread, self).__init__(parent)
        self.epoch = 0

    def init(self, data_path=None, model_path=None, learning_rate=0.0001, epoch_num=1000):
        self.train_dataset, self.val_dataset, self.test_dataset, len_dataset = dataset(data_path)
        self.train = Train(data_path, model_path, learning_rate, epoch_num, len_dataset)
        self.testTimer = QTimer(self)
        # 创建定时器
        self.testTimer.timeout.connect(self.show_time)
        # 定时超时事件绑定show_time这个函数
        self.testTimer.start(5000)
        # 定时器每一秒执行一次

    def __del__(self):
        self.wait()

    def show_time(self):
        self.epoch = self.train.return_epoch()
        self.sinOut.emit(self.epoch)

    def run(self):
        self.train(self.train_dataset, self.val_dataset, self.test_dataset)
        self.wait()