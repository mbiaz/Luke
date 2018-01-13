from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model, load_model, save_model
from keras.layers.recurrent import LSTM
from keras.callbacks import TensorBoard
from keras.optimizers import SGD, RMSprop
import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import DataLoader


class GenericModel:
    def __init__(self, dirpath,
                 batch_size, downsample_factor):  # downsample_factor= pool_size**2

        self.data_gen = DataLoader.DataLoader(dirpath=dirpath, batch_size=batch_size,
                                                        downsample_factor=downsample_factor,
                                                        max_text_len=7)
        self.data_gen.build_data()
        self.output_size = self.data_gen.get_output_size()
        img_w = self.data_gen.img_w
        img_h = self.data_gen.img_h

        if K.image_data_format() == 'channels_first':
            self.input_shape = (1, img_w, img_h)
        else:
            self.input_shape = (img_w, img_h, 1)

    @staticmethod
    def create_input(name, shape, dtype="float32"):
        return Input(name=name, shape=shape, dtype=dtype)

    @staticmethod
    def convolution_maxpooling(layer, conv_filters, kernel_size, name_conv, name_pool, pool_size, padding='same',
                               activation='relu',
                               kernel_initializer='he_normal'):
        inner = Conv2D(conv_filters, kernel_size, padding=padding,
                       activation=activation, kernel_initializer=kernel_initializer,
                       name=name_conv)(layer)
        return MaxPooling2D(pool_size=(pool_size, pool_size), name=name_pool)(inner)

    @staticmethod
    def bi_lstm(layer, h_size, name, return_sequences=True, kernel_initializer='he_normal', merge_method="add"):
        lstm_1 = LSTM(h_size, return_sequences=return_sequences, kernel_initializer=kernel_initializer, name=name)(
            layer)
        lstm_1b = LSTM(h_size, return_sequences=return_sequences, go_backwards=True,
                       kernel_initializer=kernel_initializer, name=name + 'b')(layer)
        if merge_method == "add":
            return add([lstm_1, lstm_1b])
        elif merge_method == "concatenate":
            return concatenate([lstm_1, lstm_1b])
        elif merge_method == None:
            return lstm_1, lstm_1b
        else:
            print("You must give a method in order to merge the two directional layers")
            raise Exception

    @staticmethod
    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        y_pred = y_pred[:, 2:, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    @staticmethod
    def from_conv_to_lstm_reshape(layer, name="reshape"):
        conv_to_rnn_dims = (layer.get_shape().as_list()[1],
                            (layer.get_shape().as_list()[2]) * layer.get_shape().as_list()[3])

        return Reshape(target_shape=conv_to_rnn_dims, name=name)(layer)

    @staticmethod
    def ctc_layer(y_pred, max_output_len, name_input_length, name_label, name_label_length, name_loss):
        labels = Input(name=name_label, shape=[max_output_len], dtype='float32')
        input_length = Input(name=name_input_length, shape=[1], dtype='int64')
        label_length = Input(name=name_label_length, shape=[1], dtype='int64')
        return labels, input_length, label_length, Lambda(GenericModel.ctc_lambda_func, output_shape=(1,),
                                                          name=name_loss)(
            [y_pred, labels, input_length, label_length])
    @staticmethod
    def create_tb_callbacks(tensorboard_dir):
        return TensorBoard(log_dir=tensorboard_dir, histogram_freq=0, write_graph=True,
                                 write_images=True)

    def build_model(self):
        raise NotImplementedError

    @staticmethod
    def load_model(loss, metrics, opt=RMSprop, name_model ='model.h5'):
        model = load_model(name_model, compile=False)
        return model.compile(loss=loss, optimizer=opt, metrics=metrics)




    def train(self,
              model,
              tensorboard_callback,
              loss,
              metrics,
              nb_epochs,
              save=False,
              opt=RMSprop,
              lr=0.001):

        model.compile(loss=loss, optimizer=opt(lr), metrics=metrics)
        history = model.fit_generator(generator=self.data_gen.next_batch(mode="train"),
                                      steps_per_epoch=self.data_gen.n["train"],
                                      epochs=nb_epochs,
                                      callbacks=[tensorboard_callback],
                                      validation_data=self.data_gen.next_batch(mode="test"),
                                      validation_steps=self.data_gen.n["test"])
        if save:
            model.save("model.h5")

        return model, history

    def run_model(self, loss, metrics, tensorboard_callback, nb_epochs, save=False, load=False, opt=RMSprop, lr=0.001,
                  name_model='model.h5'):
        if load :
            model = GenericModel.load_model(loss, metrics, opt, name_model)
            history = []
        else :
            model = self.build_model()
            model, history = self.train(model, tensorboard_callback, loss, metrics, nb_epochs, save, opt, lr )
        return model, history
