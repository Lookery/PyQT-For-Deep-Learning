from ui import *
import sys


if __name__ == '__main__':

    app = QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    MainWindow = MyTable()
    ChildWindow1 = TrainWin()
    ChildWindow2 = TestWin()
    ChildWindow3 = PreprocessWin()
    MainWindow.show()
    sys.exit(app.exec_())

