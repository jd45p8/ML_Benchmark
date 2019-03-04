import sys
import time
from threading import Thread
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QIcon
from main_screen_widget import Ui_Form
from SVM import *

class App(QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.title = "Benchmark"
        self.setupUi(self)
        self.init_ui()        
    def init_ui(self):
        self.setWindowTitle(self.title)
        self.start_button.clicked.connect(self.execute_benchmark)
        self.score_label.setText(str(0))
        self.state_label.hide()
        self.show()
    def start_timer(self):
        global benchmarking
        t = time.time()
        while benchmarking:
            t2 = time.time()
            self.state_label.setText("%.2f" % (t2-t) + " seg")
            time.sleep(0.1)
    def execute_benchmark(self):
        global benchmarking, svm_thread
        if benchmarking:
            benchmarking = False
            if svm_thread.isAlive():          
                svm_thread._stop()
                self.score_label.setText(0)
                print("Stoped")
            self.start_button.setText("Iniciar Prueba")
        else:
            thread_timer = Thread(target = self.start_timer)
            thread_timer.daemon = True
            self.state_label.show()
            self.start_button.setText("Detener")
            benchmarking = True

            self.score_label.setText("Datos...")
            x_train, x_test, y_train, y_test = split_data()
            self.score_label.setText("Magia...")
            svm_thread = Thread(target = svm_train_test(x_train, x_test, y_train, y_test))

            svm_thread.daemon = True

            svm_thread.start()
            thread_timer.start()

if __name__ == '__main__':
    APP = QApplication(sys.argv)
    ex = App()
    global svm_thread
    svm_thread = Thread()
    benchmarking = False
    sys.exit(APP.exec_())