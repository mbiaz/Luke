import threading
import time
import logging
import os
from random import randint


def tensdir_exists(dir):
    return os.path.exists(os.getcwd() + '/tensorboards/'+dir)


def launchTensorBoard(dir):
    while not (tensdir_exists(dir)):
        time.sleep(10)

    print("hey motherfucker")
    port = str(randint(8000, 9999))
    os.system('tensorboard --logdir=' + os.getcwd() + '/tensorboards/'+dir+ ' --port='+port)
    print("bye motherfucker")
    return
