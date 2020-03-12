#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-03-20 23:44
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import json
import random
import pickle
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
SAVES_DIR = "./models-ResNet56v2/"

# constants
IF_FAST_RUN = True
EPOCHS_OVER_NIGHT = 50
BATCH_SIZE = 15
# BATCH_SIZE = 32  # orig paper trained all networks with batch_size=128


def main():
    """ Load Config """
    with open('config.json', 'r') as f:
        CONFIG = json.load(f)
    ROOT_PATH = CONFIG["ROOT_PATH"]
    TRAIN_DATA_DIR = CONFIG["TRAIN_DATA_DIR"]
    TEST_DATA_DIR = CONFIG["TEST_DATA_DIR"]
    # MODEL_CKPT = CONFIG["MODEL_CKPT"]
    MODEL_CKPT = "ResNet56v2.11and009-auc-0.9648.h5"

    TRAIN_DATA_DIR = os.path.join(ROOT_PATH, TRAIN_DATA_DIR)
    TEST_DATA_DIR = os.path.join(ROOT_PATH, TEST_DATA_DIR)

    """ Prepare Model """
    n = 6
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
    test_filenames = os.listdir(TEST_DATA_DIR)
    test_df = pd.DataFrame({
        'filename': test_filenames
    })
    nb_samples = test_df.shape[0]

    """ Create Testing Generator """
    test_gen = ImageDataGenerator(rescale=1./255)
    test_generator = test_gen.flow_from_dataframe(
        test_df,
        TEST_DATA_DIR,
        x_col='filename',
        y_col=None,
        class_mode=None,
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        shuffle=False
    )  # Found 12500 images.

    """ Load Weights """
    model_ckpt_file = os.path.join(SAVES_DIR, MODEL_CKPT)
    if os.path.exists(model_ckpt_file):
        print("Model ckpt found! Loading...:%s" % model_ckpt_file)
        model.load_weights(model_ckpt_file)

    """ Predict """
    import time
    start = time.perf_counter()
    print("Start testing...")
    predict = model.predict_generator(
        test_generator, steps=np.ceil(nb_samples / BATCH_SIZE),
        workers=4, verbose=1)

    elapsed = (time.perf_counter() - start)
    print("Prediction time used:", elapsed)

    np.save(MODEL_TYPE + "-predict.npy", predict)

    # predict 第 1 列，是不是 bad_1
    test_df['category'] = predict[:, 0]
    print("Predict Samples: ")
    print(type(test_df))
    print(test_df.head(10))

    """ 提交submission """
    submission_df = test_df.copy()
    submission_df['id'] = submission_df['filename'].str.split('.').str[0]
    submission_df['label'] = submission_df['category']
    submission_df.drop(['filename', 'category'], axis=1, inplace=True)
    submission_df.to_csv('./submissions/submission-%s.csv' %
                         MODEL_CKPT, index=False)


if __name__ == "__main__":
    main()
