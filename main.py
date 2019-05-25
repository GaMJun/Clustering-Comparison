#! /usr/bin/python
import clusters
from warnings import filterwarnings
import threading

filterwarnings('ignore')

class ThreadDBScan(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name

    def run(self):
        print("Iniciando " + self.name)
        clusters.fbp_db()
        print("Finalizado " + self.name)


class ThreadKMeans(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name

    def run(self):
        print("Iniciando " + self.name)
        clusters.fbp_km()
        print("Finalizado " + self.name)


class ThreadAgnes(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name

    def run(self):
        print("Iniciando " + self.name)
        clusters.fbp_agnes()
        print("Finalizado " + self.name)


thread0 = ThreadDBScan("Thread-DBScan")
thread1 = ThreadKMeans("Thread-K-Means")
thread2 = ThreadAgnes("Thread-Agnes")

thread0.start()
thread1.start()
thread2.start()
