#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-03-20 23:44
# @Author  : Kelly Hwong (you@example.org)
# @Link    : http://example.org

import os
import json
import random
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split
from resnet import model_depth, resnet_v2, lr_schedule

from model import model_depth, resnet_v2, lr_schedule, binary_focal_loss
from metrics import AUC0

# Parameters we care
START_EPOCH = 0
ALPHA = 0.01
BATCH_SIZE = 16

# Training parameters
IF_FAST_RUN = False
TRAINING_EPOCHS = 150

TOTAL_TRAIN = 30000 * 0.8
TOTAL_VALIDATE = 30000 * 0.2

# constants
IF_DATA_AUGMENTATION = True
NUM_CLASSES = 2
IMAGE_WIDTH = IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 1
INPUT_SHAPE = [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS]

METRICS = [
    BinaryAccuracy(name='accuracy'),  # 整体的 accuracy
    AUC(name='auc_good_0'),  # 实际上是以 good 为 positive 的 AUC
    AUC0(name='auc_bad_1')  # 以 bad 为 positive 的 AUC
]


def main():
    print("If in eager mode: ", tf.executing_eagerly())
    print("Use tensorflow version 2.")
    assert tf.__version__[0] == "2"

    print("Load Config ...")
    with open('./config/config_linux.json', 'r') as f:
        CONFIG = json.load(f)
    ROOT_PATH = CONFIG["ROOT_PATH"]
    print(f"ROOT_PATH: {ROOT_PATH}")
    ROOT_PATH = os.path.expanduser(ROOT_PATH)
    print(f"ROOT_PATH: {ROOT_PATH}")
    TRAIN_DATA_DIR = os.path.join(ROOT_PATH, CONFIG["TRAIN_DATA_DIR"])
    print(f"TRAIN_DATA_DIR: {TRAIN_DATA_DIR}")
    TEST_DATA_DIR = os.path.join(ROOT_PATH, CONFIG["TEST_DATA_DIR"])
    print(f"TEST_DATA_DIR: {TEST_DATA_DIR}")

    print("Prepare Model")
    n = 2  # order of ResNetv2, 2 or 6
    version = 2
    depth = model_depth(n, version)
    MODEL_TYPE = 'ResNet%dv%d' % (depth, version)
    print(f"MODEL_TYPE: {MODEL_TYPE}")
    SAVES_DIR = os.path.join(ROOT_PATH, "models-%s/" % MODEL_TYPE)
    if not os.path.exists(SAVES_DIR):
        os.mkdir(SAVES_DIR)
    MODEL_CKPT = "ResNet20v2-epoch-092-auc_good_0-0.9545-auc_bad_1-0.9817.h5"
    model_ckpt_file = os.path.join(SAVES_DIR, MODEL_CKPT)
    print(f"model_ckpt_file: {model_ckpt_file}")

    model = resnet_v2(input_shape=INPUT_SHAPE, depth=depth, num_classes=2)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=lr_schedule(TRAINING_EPOCHS)),
                  metrics=METRICS)
    # model.summary()

    print("Prepare weights...")
    model_ckpt_file = MODEL_CKPT
    if os.path.isfile(model_ckpt_file):
        print("Model ckpt found! Loading...:\n%s" % model_ckpt_file)
        model.load_weights(model_ckpt_file)

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

    # predict 第 1 列，是 bad_1 的概率
    test_df['label'] = predict[:, 1]
    print("Predict Samples: ")
    print(type(test_df))
    print(test_df.head(10))

    print("Prepare submission...")
    submission_df = test_df.copy()
    submission_df['id'] = submission_df['filename'].str.split('.').str[0]
    submission_df['label'] = submission_df['label']
    submission_df.drop(['filename', 'label'], axis=1, inplace=True)
    submission_df.to_csv(
        f"./submissions/submission-{MODEL_CKPT}.csv", index=False)


if __name__ == "__main__":
    main()
