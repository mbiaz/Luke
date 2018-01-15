import os
import json
from collections import Counter
from os.path import join
import cv2
import numpy as np
from keras import backend as K
import tensorflow as tf
import random

sess = tf.Session()
K.set_session(sess)



class DataLoader:

    def __init__(self,
                 dirpath,
                 downsample_factor,
                 batch_size=8,
                 max_text_len=7):

        self.dirpath = join("datasets/",dirpath)
        self.letters = DataLoader.get_letters(self.dirpath)
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor

        img_dirpath = {'train': join(self.dirpath, 'train/images'), 'test': join(self.dirpath, 'test/images')}
        labels = {'train': json.load(open(join(self.dirpath, 'train/labels.json'), 'r')),
                  'test': json.load(open(join(self.dirpath, 'test/labels.json'), 'r'))}

        self.samples = {'train': [], 'test': []}
        for mode in ["train", "test"]:
            for filename in os.listdir(img_dirpath[mode]):
                name, ext = os.path.splitext(filename)
                if ext == '.jpg':
                    img_filepath = join(img_dirpath[mode], filename)
                    description = labels[mode][name]
                    if self.is_valid_str(description):
                        self.samples[mode].append([img_filepath, description])
        self.n = {"train": len(self.samples["train"]), "test": len(self.samples["test"])}
        self.indexes = {"train": list(range(self.n["train"])), "test": list(range(self.n["test"]))}
        self.cur_index = {"train": 0, "test": 0}
        self.get_images_dimensions()

    @staticmethod
    def get_counter(dirpath):
        description = json.load(open(dirpath, 'r'))
        text = ""
        for idx in description:
            text += description[idx]

        return Counter(text)

    def get_images_dimensions(self):
        dimensions = np.zeros((self.n["train"] + self.n["test"], 3), dtype=np.int)
        k = 0
        for mode in ["train", "test"]:
            for i, (img_filepath, text) in enumerate(self.samples[mode]):
                img = cv2.imread(img_filepath)
                img_dimensions = img.shape
                dimensions[k, :] = img_dimensions
                k += 1
        if ((dimensions - dimensions[0]) == 0).all():
            print("the images are all of the same dimension : ", dimensions[0])
            self.img_h, self.img_w, self.channels = dimensions[0]
        else:
            print("the images are no all of the same dimension")
            raise Exception

    @staticmethod
    def get_letters(dirpath):
        c_train = DataLoader.get_counter(dirpath + "/train/labels.json")
        c_test = DataLoader.get_counter(dirpath + "/test/labels.json")
        letters_test = set(c_test.keys())
        letters_train = set(c_train.keys())
        let = sorted(list(letters_train))
        if letters_train == letters_test:
            print('Letters in train and test do match')
        else:
            raise Exception()
        return let

    def get_output_size(self):
        return len(self.letters) + 1

    def labels_to_text(self, labels):
        return ''.join(list(map(lambda x: self.letters[int(x)], labels)))

    def text_to_labels(self, text):
        return list(map(lambda x: self.letters.index(x), text))

    def is_valid_str(self, s):
        for ch in s:
            if not ch in self.letters:
                return False
        return True

    def build_data(self):
        self.imgs = {"train": np.zeros((self.n["train"], self.img_h, self.img_w)),
                     "test": np.zeros((self.n["test"], self.img_h, self.img_w))}
        self.texts = {"train": [], "test": []}
        for mode in ["train", "test"]:
            for i, (img_filepath, text) in enumerate(self.samples[mode]):
                img = cv2.imread(img_filepath)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (self.img_w, self.img_h))
                img = img.astype(np.float32)
                img /= 255
                self.imgs[mode][i, :, :] = img
                self.texts[mode].append(text)

    def next_sample(self, mode):
        self.cur_index[mode] += 1
        if self.cur_index[mode] >= self.n[mode]:
            self.cur_index[mode] = 0
            random.shuffle(self.indexes[mode])
        return self.imgs[mode][self.indexes[mode][self.cur_index[mode]]], self.texts[mode][
            self.indexes[mode][self.cur_index[mode]]]

    def next_batch(self, mode, batch_size):
        self.batch_size = batch_size
        while True:
            if K.image_data_format() == 'channels_first':
                X_data = np.ones([self.batch_size, 1, self.img_w, self.img_h])
            else:
                X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])

            Y_data = np.ones([self.batch_size, self.max_text_len])
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)
            label_length = np.zeros((self.batch_size, 1))

            for i in range(self.batch_size):
                img, text = self.next_sample(mode)
                img = img.T
                if K.image_data_format() == 'channels_first':
                    img = np.expand_dims(img, 0)
                else:
                    img = np.expand_dims(img, -1)
                X_data[i] = img
                Y_data[i] = self.text_to_labels(text)
                label_length[i] = len(text)

            inputs = {
                'the_input': X_data,
                'the_labels': Y_data,
                'input_length': input_length,
                'label_length': label_length,
            }
            outputs = {'ctc': np.zeros([self.batch_size])}
            yield (inputs, outputs)
