#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-01-20 06:51
# @Author  : Your Name (you@example.org)
# @Link    : https://stackoverflow.com/questions/43702323/how-to-load-only-specific-weights-on-keras

import os
import random
import numpy as np
import pandas as pd
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split
from resnet import model_depth, resnet_v2, lr_schedule
from model import auc

# Training parameters
IF_DATA_AUGMENTATION = True
NUM_CLASSES = 2
IMAGE_WIDTH = IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 1
INPUT_SHAPE = [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS]
# data path
TRAIN_DATA_DIR = "./data/train/"
TEST_DATA_DIR = "./data/test/all_tests"

# constants
IF_FAST_RUN = True
EPOCHS_OVER_NIGHT = 50
BATCH_SIZE = 15
# BATCH_SIZE = 32  # orig paper trained all networks with batch_size=128


class ResNetParam(object):
    def __init__(self, n=2, version=2):
        self.n = n
        self.version = version
        self.depth = model_depth(self.n, self.version)
        self.model_type = 'ResNet%dv%d' % (self.depth, self.version)
        self.saves_dir = "./models-%s/" % self.model_type


def main():
    """ Prepare Model """
    src_param = ResNetParam(n=2, version=2)
    dest_param = ResNetParam(n=6, version=2)
    if not os.path.exists(src_param.saves_dir):
        print("Source modle path %s not exists, creating..." %
              src_param.saves_dir)
        os.mkdir(src_param.saves_dir)
    if not os.path.exists(dest_param.saves_dir):
        print("Destination modle path %s not exists, creating..." %
              dest_param.saves_dir)
        os.mkdir(dest_param.saves_dir)

    """ Resume Src Model """
    print("Src model type: %s" % src_param.model_type)
    src_model = resnet_v2(input_shape=INPUT_SHAPE,
                          depth=src_param.depth, num_classes=2)
    # src_model.summary()

    """ Loading Src Weights """
    MODEL_CKPT_FILE = "ResNet20v2.020-auc-0.9736.h5"
    filepath = os.path.join(src_param.saves_dir, MODEL_CKPT_FILE)
    print("loading weights from: %s..." % filepath)
    src_model.load_weights(filepath)
    src_weights_list = src_model.get_weights()

    """ Show src weights """
    """
    for i in range(len(src_model.layers)):
        print("Printing layer: %d" % i, src_model.layers[i])
        weights = src_model.layers[i].get_weights()
        for weight in weights:  # Layer type
            print(weight.shape)
    input()
    """

    """ Create Dest Model """
    dest_model = resnet_v2(input_shape=INPUT_SHAPE,
                           depth=dest_param.depth, num_classes=2)
    # dest_model.summary()
    print("Dest model type: %s" % dest_param.model_type)
    # input不算
    # layer 1-24 to 1-24
    for i in range(1, 24):
        dest_model.layers[i].set_weights(src_model.layers[i].get_weights())
    print("Partially load weights from layer 1-24 successfully!")

    # layer 25-45 to 65-85
    for i in range(25, 45):
        dest_model.layers[i+40].set_weights(src_model.layers[i].get_weights())
    print("Partially load weights from layer 25-45 successfully!")

    # layer 46-65 to 126-145
    for i in range(46, 65):
        dest_model.layers[i+80].set_weights(src_model.layers[i].get_weights())
    print("Partially load weights from layer 46-65 successfully!")

    # 69 to 189
    dest_model.layers[69+120].set_weights(src_model.layers[69].get_weights())
    print("Partially load weights from layer 69 successfully!")

    """ Show dest weights """
    for i in range(len(dest_model.layers)):
        print("Printing layer: %d" % i, dest_model.layers[i])
        weights = dest_model.layers[i].get_weights()
        for weight in weights:  # Layer type
            print(weight.shape)

    print("Saving new model...")
    filename = "%s.h5" % dest_param.model_type
    dest_model.save_weights(filename)


if __name__ == "__main__":
    main()
