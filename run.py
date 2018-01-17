import sys, os
from importlib import import_module
import threading
from utils import launchTensorBoard


def getopts(argv):
    argv = argv[1:]
    opts = {}
    while argv:
        index = argv[0].find('=')
        opts[argv[0][2:index]] = argv[0][index + 1:]
        argv = argv[1:]
    return opts["model"], opts["dataset"], int(opts["save"]), int(opts["load"])


if __name__ == '__main__':
    MODEL, DATASET, SAVE, LOAD = getopts(sys.argv)
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/models")
    t_tensorboard = threading.Thread(target=launchTensorBoard, args=([DATASET+'/'+MODEL]))
    t_tensorboard.start()
    mod = import_module(MODEL)
    model = mod.MyModel(dirpath=DATASET)
    model.run_model(save=SAVE,
                    load=LOAD)
    os.kill(os.getpid(),1)
