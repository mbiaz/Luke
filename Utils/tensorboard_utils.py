import threading
import time
import logging
import os


def tensdir_exists():
    return os.path.exists(os.getcwd()+'/tensorboard')


def launchTensorBoard():
    print("****************")
    print("ENCULE ")
    print("****************")
    while not(tensdir_exists()):
        print("****************")
        print("WAIT MOTHERFUCKER")
        print(os.getcwd()+'/tensorboard')
        print("****************")
        time.sleep(10)

    os.system('tensorboard --logdir='+os.getcwd()+'/tensorboard'+' --port=8082')
    return
