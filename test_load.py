from keras.models import load_model
import sys
import os
from importlib import import_module
from keras.optimizers import RMSprop
import numpy as np
import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt
import cv2

sess = tf.Session()
K.set_session(sess)

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/models")
mod = import_module("model_lstm")
loader_data = import_module("load_dataset")



model = load_model('model.h5', custom_objects={'<lambda>': lambda y_true, y_pred:y_pred})
model.compile(optimizer=RMSprop(0.001), metrics=["binary_accuracy"], loss={'ctc': lambda y_true, y_pred:y_pred})
batch_size = 12
data = loader_data.TextImageGenerator(dirpath = "dataset_6",
                                      batch_size = batch_size ,
                                      downsample_factor = 4,
                                      max_text_len = 7)



data.build_data()

net_inp = model.get_layer(name= "the_input").input
net_out = model.get_layer(name="dense2").output


i = 4
for input, _ in data.next_batch(mode="test"):
    X_data = input["the_input"][i][np.newaxis, :]
    net_out_val = sess.run(net_out, feed_dict={net_inp: X_data})
    label=input["the_labels"][i]
    text = ''.join(map(lambda x : data.letters[int(x)], label))
    print(text)
    st = ""
    for t in net_out_val[0]:
        val = np.argmax(t)
        if str(val) != "16":
            st+=data.letters[int(val)]
    print(st)
    print(input["the_input"][i])
    print(input["the_input"][i].shape)
    img =  cv2.cvtColor( input["the_input"][i][:,np.newaxis]*255, cv2.COLOR_BGR2GRAY)
    cv2.imshow("aze", img)
    cv2.waitKey(0)
    break
