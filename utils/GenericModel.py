import os
import json
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model, load_model, save_model
from keras.layers.recurrent import LSTM
from keras.callbacks import TensorBoard
from keras.optimizers import SGD, RMSprop
from utils.DataLoader import DataLoader


class GenericModel:
    def __init__(self, dirpath):
        self.optimizer = None
        self.learning_rate = None
        self.batch_size = None
        self.metrics = None
        self.loss = None
        self.epochs = None
        self.dirpath = dirpath
        self.tensorboard = self.create_tb_callbacks("./tensorboards/"+self.dirpath+'/'+type(self).__name__)
        self.name_model = os.getcwd() + '/saved_models/'+self.dirpath+'/'+type(self).__name__+".h5"
        self.data_gen = DataLoader(dirpath=dirpath, batch_size=self.batch_size,
                                                        downsample_factor=0)
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
    def ctc_loss():
        return {'ctc': lambda y_true, y_pred: y_pred}
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
    def load_model(loss, metrics, opt, name_model):
        model = load_model(name_model, compile=False)
        return model.compile(loss=loss, optimizer=opt, metrics=metrics)

    def initialize_training(self):
        raise NotImplementedError

    def train(self,
              model,
              tensorboard_callback,
              loss,
              metrics,
              nb_epochs,
              save,
              opt,
              lr):

        model.compile(loss=loss, optimizer=opt(lr), metrics=metrics)
        history = model.fit_generator(generator=self.data_gen.next_batch(mode="train", batch_size=self.batch_size),
                                      steps_per_epoch=self.data_gen.n["train"],
                                      epochs=nb_epochs,
                                      callbacks=[tensorboard_callback],
                                      validation_data=self.data_gen.next_batch(mode="test", batch_size=self.batch_size),
                                      validation_steps=self.data_gen.n["test"])
        if save:
            print("saving model into : ")

            if not os.path.exists(os.getcwd() + '/saved_models/'+self.dirpath):
                os.makedirs(os.getcwd() + '/saved_models/'+self.dirpath)
            if os.path.exists(os.getcwd() + '/saved_models/'+self.dirpath+'/'+type(self).__name__+".h5") :
                print("model already saved a long time ago ")
                raise Exception
            model.save(os.getcwd() + '/saved_models/'+self.dirpath+'/'+type(self).__name__+".h5")
        return model, history

    def run_model(self, save=False, load=False):

        try:
            self.initialize_training()
        except Exception:
            print("you need to over-load the method initialize_training in your model ")
            raise Exception

        if self.optimizer is None:
            print("please provide an optimizer")
            raise Exception

        if self.learning_rate is None:
            print("please provide a learning_rate")
            raise Exception

        if self.metrics is None:
            print("please provide metrics")
            raise Exception

        if self.loss is None:
            print("please provide a loss function")
            raise Exception

        if self.epochs is None:
            print("please provide a number of epochs")
            raise Exception




        if load:
            print("Loading model")
            model = GenericModel.load_model(loss=self.loss,
                                            metrics=self.metrics,
                                            opt=self.optimizer,
                                            name_model=self.name_model)
            history = []
        else:
            model = self.build_model()
            with open(os.getcwd() + '/summaries/'+type(self).__name__+'.txt','w') as fh:
                model.summary(print_fn=lambda x: fh.write(x + '\n'))


            model, history = self.train(model=model,
                                        tensorboard_callback=self.tensorboard,
                                        loss=self.loss,
                                        metrics=self.metrics,
                                        nb_epochs=self.epochs,
                                        save=save,
                                        opt=self.optimizer,
                                        lr=self.learning_rate)
            with open(os.getcwd()+'/logs/'+type(self).__name__+'.json','w') as log_file:
                log = {}
                log["name"] = type(self).__name__
                log["batch_size"] = self.batch_size
                log["optimizer"]= self.optimizer.__name__
                log["learning_rate"] = self.learning_rate
                log["epochs"] = self.epochs
                log["nb_train"]= self.data_gen.n["train"]
                log["nb_test"]= self.data_gen.n["test"]
                log["data_dim"]= [int(self.input_shape[0]), int(self.input_shape[1]),int( self.input_shape[2])]
                var_log = {}
                for keys_indicators in history.history.keys():
                    var_log[keys_indicators] = history.history[keys_indicators]
                print(var_log)
                log["train"]={}
                for i in range(self.epochs):
                    log["train"][str(i)]= {}
                    for keys_indicators in history.history.keys():
                        log["train"][str(i)][keys_indicators] = var_log[keys_indicators][i]

                log["max_values"] = {}
                for keys_indicators in history.history.keys():
                    log["max_values"][keys_indicators] = sorted(var_log[keys_indicators])[-1]
                json_string = model.to_json()
                log["summary"]=json_string
                json.dump(log, log_file, indent=4)

        return model, history
