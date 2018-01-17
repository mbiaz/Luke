import threading
import time
import logging
import os


def tensdir_exists(dir):
    return os.path.exists(os.getcwd() + '/tensorboards/'+dir)


def launchTensorBoard(dir):
    while not (tensdir_exists(dir)):
        time.sleep(10)

    print("hey motherfucker")
    os.system('tensorboard --logdir=' + os.getcwd() + '/tensorboards/'+dir+ ' --port=8083')
    print("bye motherfucker")
    return
