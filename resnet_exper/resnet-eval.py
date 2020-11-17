#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Mar-14-20 18:53
# @Author  : Kelly Hwong (you@example.org)
# @Link    : http://example.org

import os
import json
import random
import pickle
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras  # tf2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator, load_img

from resnet import model_depth, resnet_v2, lr_schedule
from metrics import AUC

# Evaluation parameters
START_EPOCH = 0
IF_FAST_RUN = False
TRAINING_EPOCHS = 50

TOTAL_TRAIN = 30000 * 0.8
TOTAL_VALIDATE = 30000 * 0.2

N_VALIDATION = 6000
N_VAL_GOOD = N_VALIDATION * 0.9
N_VAL_BAD = N_VALIDATION * 0.1

# constants
IF_DATA_AUGMENTATION = True
NUM_CLASSES = 2
IMAGE_WIDTH = IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 1
INPUT_SHAPE = [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS]

METRICS = [
    keras.metrics.BinaryAccuracy(name='accuracy'),
    AUC(name='auc_good_0')
    # AUC(name='auc_bad_1') # 以 bad 为 positive 的 AUC
]


def main():
    """ Use tensorflow version 2 """
    assert tf.__version__[0] == "2"

    """ Load Config """
    with open('./config/config.json', 'r') as f:
        CONFIG = json.load(f)
    BATCH_SIZE = 32  # CONFIG["BATCH_SIZE"]
    ROOT_PATH = CONFIG["ROOT_PATH"]
    TRAIN_DATA_DIR = CONFIG["TRAIN_DATA_DIR"]
    VALIDATION_DATA_DIR = CONFIG["VALIDATION_DATA_DIR"]
    TRAIN_DATA_DIR = os.path.join(ROOT_PATH, TRAIN_DATA_DIR)
    VALIDATION_DATA_DIR = os.path.join(ROOT_PATH, VALIDATION_DATA_DIR)
    MODEL_CKPT = CONFIG["MODEL_CKPT"]

    """ Prepare Model """
    n = 6  # order of ResNetv2
    version = 2
    depth = model_depth(n, version)
    MODEL_TYPE = 'ResNet%dv%d' % (depth, version)
    SAVES_DIR = "models-%s/" % MODEL_TYPE
    SAVES_DIR = os.path.join(ROOT_PATH, SAVES_DIR)
    if not os.path.exists(SAVES_DIR):
        os.mkdir(SAVES_DIR)
    model = resnet_v2(input_shape=INPUT_SHAPE, depth=depth, num_classes=2)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=lr_schedule(TRAINING_EPOCHS)),
                  metrics=METRICS)
    # model.summary()
    print(MODEL_TYPE)

    """ Prepare Testing Data """
    val_filenames = os.listdir(os.path.join(VALIDATION_DATA_DIR, "bad_1"))
    val_bad_df = pd.DataFrame({
        'filename': val_filenames
    })
    n_bad_samples = val_bad_df.shape[0]

    """ Prepare good samples """
    val_filenames = os.listdir(os.path.join(VALIDATION_DATA_DIR, "good_0"))
    val_good_df = pd.DataFrame({
        'filename': val_filenames
    })
    n_good_samples = val_good_df.shape[0]

    """ Create bad sample validation generator """
    valid_bad_datagen = ImageDataGenerator(rescale=1./255)
    valid_bad_generator = valid_bad_datagen.flow_from_dataframe(
        val_bad_df,
        os.path.join(VALIDATION_DATA_DIR, "bad_1"),
        x_col='filename',
        y_col=None,
        class_mode=None,
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    """ Create good sample validation generator """
    valid_good_datagen = ImageDataGenerator(rescale=1./255)
    valid_good_generator = valid_good_datagen.flow_from_dataframe(
        val_good_df,
        os.path.join(VALIDATION_DATA_DIR, "good_0"),
        x_col='filename',
        y_col=None,
        class_mode=None,
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    """ Load Weights """
    model_ckpt_file = os.path.join(SAVES_DIR, MODEL_CKPT)
    if os.path.exists(model_ckpt_file):
        print("Model ckpt found! Loading...:%s" % model_ckpt_file)
        model.load_weights(model_ckpt_file)

    """ Predict """
    import time
    start = time.perf_counter()
    # print("Start validating bad samples...")
    print("Start validating samples...")
    predict = model.predict_generator(
        valid_good_generator, steps=np.ceil(N_VAL_GOOD / BATCH_SIZE),
        workers=4, verbose=1)

    elapsed = (time.perf_counter() - start)
    print("Prediction time used:", elapsed)

    np.save(MODEL_TYPE + "-predict.npy", predict)

    # predict 第 1 列，是 bad_1 的概率
    val_good_df['predict'] = predict[:, 0]
    print("Predict Samples: ")
    print(type(val_good_df))
    print(val_good_df.head(10))

    """ Submit prediction """
    submission_df = val_good_df.copy()
    # submission_df.drop(['filename', 'predict'], axis=1, inplace=True)
    submission_df.to_csv('./submissions/submission-%s.csv' %
                         MODEL_CKPT, index=False)


if __name__ == "__main__":
    main()
