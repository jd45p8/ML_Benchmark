import sys
import time
from multiprocessing import Process
from threading import Thread
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QIcon
from main_screen_widget import Ui_Form
from Learning import Training_Object

class App(QWidget, Ui_Form):
    #Constructor
    def __init__(self):
        super().__init__()
        self.title = "Benchmark"
        self.setupUi(self)
        self.init_ui()
    #Initializing the Graphical Interface       
    def init_ui(self):
        self.setWindowTitle(self.title)
        self.start_button.clicked.connect(self.execute_benchmark)
        self.score_label.setText(str(0))
        self.state_label.hide()
        self.show()
    #Timer for time measurement while the training is takig place
    def start_timer(self):
        global benchmarking, training_process, train_o
        t = time.time()
        while training_process.is_alive():
            t2 = time.time()
            self.state_label.setText("%.2f" % (t2-t) + " seg")
            time.sleep(0.1)
        if benchmarking:
            self.score_label.setText("%.3f" % ((t2-t)/train_o.sample_count))
            self.start_button.setText("Repetir prueba")
            benchmarking = False
    #Handler for bencmark execution
    def execute_benchmark(self):
        global benchmarking, training_process, train_o
        if benchmarking:
            benchmarking = False
            if training_process.is_alive():
                training_process.terminate()
                self.score_label.setText("0")
            self.start_button.setText("Iniciar Prueba")
        else:
            timer_thread = Thread(target = self.start_timer)
            timer_thread.daemon = True
            self.state_label.show()
            self.start_button.setText("Detener")
            benchmarking = True

            self.score_label.setText("Datos...")
            train_o = Training_Object()
            train_o.split_data()
            self.score_label.setText("Magia...")
            training_process = Process(target = train_o.train_test)

            training_process.daemon = True

            training_process.start()
            timer_thread.start()

if __name__ == '__main__':
    APP = QApplication(sys.argv)
    ex = App()
    benchmarking = False
    sys.exit(APP.exec_())