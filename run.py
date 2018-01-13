import sys, os
from importlib import import_module
from keras.optimizers import SGD, RMSprop
import threading

MODEL_NAME = sys.argv[1]
DATASET_DIR = sys.argv[2]
SAVE = sys.argv[3]
LOAD = sys.argv[4]
TENSORBOARD = sys.argv[5]



sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/models")
mod = import_module(MODEL_NAME)
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/Utils")
import tensorboard_utils

# t_tensorboard = threading.Thread(target=tensorboard_utils.launchTensorBoard, args=([]))
# t_tensorboard .start()

model = mod.MyModel(dirpath=DATASET_DIR,
                    batch_size=8,
                    downsample_factor=4)


model.run_model(loss={'ctc': lambda y_true, y_pred: y_pred},
                metrics=["binary_accuracy"],
                tensorboard_callback=model.create_tb_callbacks("./tensorboard"),
                nb_epochs=1,
                save=False,
                load=False,
                opt=RMSprop,
                lr=0.001,
                name_model='model.h5')

print('done')
