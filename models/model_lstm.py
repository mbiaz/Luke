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
        self.img_w = self.data_gen.img_w
        self.img_h = self.data_gen.img_h

        if K.image_data_format() == 'channels_first':
            self.input_shape = (1, self.img_w, self.img_h)
        else:
            self.input_shape = (self.img_w, self.img_h, 1)

    def train(self,
                    loss={'ctc': lambda y_true, y_pred: y_pred},
                    load=False,
                    save=False,
                    nb_epochs=3,
                    lr=0.001):

        def ctc_lambda_func(args):
            y_pred, labels, input_length, label_length = args
            y_pred = y_pred[:, 2:, :]
            return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

        # model
        conv_filters = 16
        kernel_size = (3, 3)
        pool_size = 2
        rnn_size = 512

        act = 'relu'
        input_data = Input(name='the_input', shape=self.input_shape, dtype='float32')
        inner = Conv2D(conv_filters, kernel_size, padding='same',
                       activation=act, kernel_initializer='he_normal',
                       name='conv1')(input_data)
        inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
        inner = Conv2D(conv_filters, kernel_size, padding='same',
                       activation=act, kernel_initializer='he_normal',
                       name='conv2')(inner)
        inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

        conv_to_rnn_dims = (self.img_w // (pool_size ** 2), (self.img_h // (pool_size ** 2)) * conv_filters)
        inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

        lstm_1 = LSTM(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='lstm_1')(inner)
        lstm_1b = LSTM(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal',
                       name='lstm_1b')(
            inner)
        lstm1_merged = add([lstm_1, lstm_1b])
        lstm_2 = LSTM(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
        lstm_2b = LSTM(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal',
                       name='lstm_2b')(
            lstm1_merged)

        inner = Dense(self.output_size, kernel_initializer='he_normal',
                      name='dense2')(concatenate([lstm_2, lstm_2b]))
        y_pred = Activation('softmax', name='softmax')(inner)
        Model(inputs=input_data, outputs=y_pred).summary()

        labels = Input(name='the_labels', shape=[self.data_gen.max_text_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

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
