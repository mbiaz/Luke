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
import load_dataset


class GenericModel:
    def __init__(self, dirpath,
                 batch_size=8, downsample_factor=4):  # downsample_factor= pool_size**2

        self.data_gen = load_dataset.TextImageGenerator(dirpath=dirpath, batch_size=batch_size,
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
        if merge_method=="add":
            return add([lstm_1, lstm_1b])
        elif merge_method == "concatenate":
            return concatenate([lstm_1, lstm_1b])
        elif merge_method == None :
            return lstm_1, lstm_1b
        else :
            print("You must give a method in order to merge the two directional layers")
            raise Exception


    @staticmethod
    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        y_pred = y_pred[:, 2:, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)



    def train(self,
              loss={'ctc': lambda y_true, y_pred: y_pred},
              load=False,
              save=False,
              nb_epochs=3,
              lr=0.001):

        # model
        conv_filters = 16
        kernel_size = (3, 3)
        pool_size = 2
        rnn_size = 512

        # the input layer
        input_data = Input(name='the_input', shape=self.input_shape, dtype='float32')

        # the convolutional layers
        conv1 = GenericModel.convolution_maxpooling(input_data, conv_filters, kernel_size, 'conv1', 'max1', pool_size)
        conv2 = GenericModel.convolution_maxpooling(conv1, conv_filters, kernel_size, 'conv2', 'max2', pool_size)

        # reshape the last output of the convolution in order to be the input of the rnn
        conv_to_rnn_dims = (conv2.get_shape().as_list()[1],
                            (conv2.get_shape().as_list()[2]) * conv2.get_shape().as_list()[3])
        inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(conv2)

        # the lstm layers
        lstm1 = GenericModel.bi_lstm(inner, rnn_size, 'lstm_1', merge_method="add")
        lstm2 = GenericModel.bi_lstm(lstm1, rnn_size, 'lstm_2',merge_method="concatenate")

        # one last dense layer
        y_pred = Dense(self.output_size, kernel_initializer='he_normal', name='dense2', activation='softmax')(lstm2)

        #inner = Dense(self.output_size, kernel_initializer='he_normal', name='dense2')(lstm2_merged)
        #y_pred = Activation('softmax', name='softmax')(inner)

        labels = Input(name='the_labels', shape=[self.data_gen.max_text_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        loss_out = Lambda(GenericModel.ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

        sgd = RMSprop(lr=lr)
        if load:
            model = load_model('model.h5', compile=False)
        else:
            model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

        model.compile(loss=loss, optimizer=sgd, metrics=["binary_accuracy"])

        if not load:

            tbCallBack = TensorBoard(log_dir='./tensorboard', histogram_freq=0, write_graph=True,
                                     write_images=True)

            history = model.fit_generator(generator=self.data_gen.next_batch(mode="train"),
                                          steps_per_epoch=self.data_gen.n["train"],
                                          epochs=nb_epochs,
                                          callbacks=[tbCallBack],
                                          validation_data=self.data_gen.next_batch(mode="test"),
                                          validation_steps=self.data_gen.n["test"])
            if save:
                model.save("model.h5")

        return model, history
