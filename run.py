import sys, os
from importlib import import_module
import threading

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/models")
mod = import_module(sys.argv[1])
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/Utils")
import tensorboard_utils

# t_tensorboard = threading.Thread(target=tensorboard_utils.launchTensorBoard, args=([]))
# t_tensorboard .start()

model = mod.GenericModel(dirpath=sys.argv[2],
                         batch_size=8,
                         downsample_factor=4)

model_, history_ = model.train(
                                load=False,
                                save=False,
                                nb_epochs=3,
                                lr=0.001)
