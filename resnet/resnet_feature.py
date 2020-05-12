from __future__ import print_function
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Mar-15-20 22:02
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import json
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras  # tf2
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator, load_img

from resnet import model_depth, resnet_v2, lr_schedule
# from model import auc # tf1
from metrics import AUC

# Training parameters
START_EPOCH = 0
IF_FAST_RUN = False
TRAINING_EPOCHS = 50

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
    keras.metrics.BinaryAccuracy(name='accuracy'),
    AUC(name='auc_good_0')
    # AUC(name='auc_bad_1') # 以 bad 为 positive 的 AUC
]


def main():
    """ Use tensorflow version 2 """
    # assert tf.__version__[0] == "2"

    """ Load Config """
    with open('./config/config_origin.json', 'r') as f:
        CONFIG = json.load(f)
    BATCH_SIZE = CONFIG["BATCH_SIZE"]
    ROOT_PATH = CONFIG["ROOT_PATH"]
    TRAIN_DATA_DIR = CONFIG["TRAIN_DATA_DIR"]
    TEST_DATA_DIR = CONFIG["TEST_DATA_DIR"]
    TRAIN_DATA_DIR = os.path.join(ROOT_PATH, TRAIN_DATA_DIR)
    TEST_DATA_DIR = os.path.join(ROOT_PATH, TEST_DATA_DIR)
    MODEL_CKPT = CONFIG["MODEL_CKPT"]

    """ Prepare Model """
    n = 6  # order of ResNetv2
    version = 2
    depth = model_depth(n, version)
    MODEL_TYPE = 'ResNet%dv%d' % (depth, version)
    SAVES_DIR = "models-%s/" % MODEL_TYPE
    SAVES_DIR = os.path.join(ROOT_PATH, SAVES_DIR)
    MODEL_CKPT = os.path.join(SAVES_DIR, MODEL_CKPT)

    # Features directory
    FEATURE_DIR = os.path.join(ROOT_PATH, "features")
    FEATURE_DIR = os.path.join(FEATURE_DIR, "models-%s/" % MODEL_TYPE)
    if not os.path.exists(FEATURE_DIR):
        os.mkdir(FEATURE_DIR)

    if not os.path.exists(SAVES_DIR):
        os.mkdir(SAVES_DIR)
    model = resnet_v2(input_shape=INPUT_SHAPE, depth=depth, num_classes=2)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=lr_schedule(TRAINING_EPOCHS)),
                  metrics=METRICS)
    # model.summary()
    print(MODEL_TYPE)

    """ Load Weights """
    model_ckpt_file = os.path.join(SAVES_DIR, MODEL_CKPT)
    if os.path.exists(model_ckpt_file):
        print("Model ckpt found! Loading...:%s" % model_ckpt_file)
        model.load_weights(model_ckpt_file)

    """ Extract Testing Data """
    _train_filenames = os.listdir(os.path.join(TRAIN_DATA_DIR, "bad_1"))
    train_bad_df = pd.DataFrame({
        'filename': _train_filenames
    })
    n_bad_samples = train_bad_df.shape[0]
    train_bad_df.to_csv(os.path.join(
        FEATURE_DIR, "bad_samples_list.csv"), index=False)

    """ Extract good samples """
    _train_filenames = os.listdir(os.path.join(TRAIN_DATA_DIR, "good_0"))
    train_good_df = pd.DataFrame({
        'filename': _train_filenames
    })
    n_good_samples = train_good_df.shape[0]
    train_good_df.to_csv(os.path.join(
        FEATURE_DIR, "good_samples_list.csv"), index=False)

    """ Create bad sample validation generator """
    train_bad_datagen = ImageDataGenerator(rescale=1./255)
    train_bad_generator = train_bad_datagen.flow_from_dataframe(
        train_bad_df,
        os.path.join(TRAIN_DATA_DIR, "bad_1"),
        x_col='filename',
        y_col=None,
        class_mode=None,
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    """ Create bad sample validation generator """
    train_good_datagen = ImageDataGenerator(rescale=1./255)
    train_good_generator = train_good_datagen.flow_from_dataframe(
        train_good_df,
        os.path.join(TRAIN_DATA_DIR, "good_0"),
        x_col='filename',
        y_col=None,
        class_mode=None,
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    """ Extractor """
    extractor = Model(
        model.inputs, model.layers[-2].output)  # flatten_2 (Flatten) (None, 12544)
    # features = extractor.predict(data)

    """ Extract train set 的特征 """
    import time
    # bad samples
    start = time.perf_counter()
    print("Start extracting bad samples...")
    features = extractor.predict_generator(
        train_bad_generator, steps=np.ceil(n_bad_samples / BATCH_SIZE),
        workers=4, verbose=1)
    print("features.shape:", features.shape)  # (16/32/etc, 12544)
    np.save(os.path.join(FEATURE_DIR, "features_train_bad.npy"), features)

    elapsed = (time.perf_counter() - start)
    print("Prediction time used:", elapsed)
    # TODO 用 pandas 存储
    # good samples
    start = time.perf_counter()
    print("Start extracting good samples...")
    features = extractor.predict_generator(
        train_good_generator, steps=np.ceil(n_good_samples / BATCH_SIZE),
        workers=4, verbose=1)
    print("features.shape:", features.shape)  # (16/32/etc, 12544)
    np.save(os.path.join(FEATURE_DIR, "features_train_good.npy"), features)

    elapsed = (time.perf_counter() - start)
    print("Prediction time used:", elapsed)


if __name__ == "__main__":
    main()
