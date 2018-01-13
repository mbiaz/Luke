import sys, os
from keras.layers import Input, Dense, Activation
from keras.models import Model, load_model, save_model
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/utils")
import GenericModel

class MyModel(GenericModel.GenericModel):


    def build_model(self):
        conv_filters = 16
        kernel_size = (3, 3)
        pool_size = 2
        rnn_size = 512

        input_data = MyModel.create_input(name='the_input', shape=self.input_shape, dtype='float32')

        conv1 = MyModel.convolution_maxpooling(input_data, conv_filters, kernel_size, 'conv1', 'max1', pool_size)
        conv2 = MyModel.convolution_maxpooling(conv1, conv_filters, kernel_size, 'conv2', 'max2', pool_size)

        reshaped_layer = MyModel.from_conv_to_lstm_reshape(conv2, name="reshape")

        lstm1 = MyModel.bi_lstm(reshaped_layer, rnn_size, 'lstm_1', merge_method="add")
        lstm2 = MyModel.bi_lstm(lstm1, rnn_size, 'lstm_2', merge_method="concatenate")

        y_pred = Dense(self.output_size, kernel_initializer='he_normal', name='dense2', activation='softmax')(lstm2)

        labels, input_length, label_length, loss_out = MyModel.ctc_layer(y_pred=y_pred,
                                                                         max_output_len=self.data_gen.max_text_len,
                                                                         name_input_length='input_length',
                                                                         name_label='the_labels',
                                                                         name_label_length='label_length',
                                                                         name_loss='ctc')

        return Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
