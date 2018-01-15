import threading
import time
import logging
import os


def tensdir_exists():
    return os.path.exists(os.getcwd() + '/tensorboard')


def launchTensorBoard():
    while not (tensdir_exists()):
        time.sleep(10)

    print("hey motherfucker")
    os.system('tensorboard --logdir=' + os.getcwd() + '/tensorboard' + ' --port=8083')
    print("bye motherfucker")
    return
