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

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.metrics import AUC, BinaryAccuracy

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
    with open('./config/config_origin.json', 'r') as f:
        CONFIG = json.load(f)
    ROOT_PATH = CONFIG["ROOT_PATH"]
    TRAIN_DATA_DIR = os.path.join(ROOT_PATH, CONFIG["TRAIN_DATA_DIR"])

    print("Prepare Model")
    n = 2  # order of ResNetv2, 2 or 6
    version = 2
    depth = model_depth(n, version)
    MODEL_TYPE = 'ResNet%dv%d' % (depth, version)
    print(f"MODEL_TYPE: {MODEL_TYPE}")
    SAVES_DIR = os.path.join(ROOT_PATH, "models-%s/" % MODEL_TYPE)
    MODEL_CKPT = os.path.join(SAVES_DIR, CONFIG["MODEL_CKPT"])

    if not os.path.exists(SAVES_DIR):
        os.mkdir(SAVES_DIR)
    model = resnet_v2(input_shape=INPUT_SHAPE, depth=depth, num_classes=2)
    model.compile(loss=[binary_focal_loss(alpha=ALPHA, gamma=1)],
                  optimizer=Adam(learning_rate=lr_schedule(epoch=0)),
                  metrics=METRICS)
    # model.summary()

    print("Resume Training...")
    model_ckpt_file = MODEL_CKPT
    if os.path.isfile(model_ckpt_file):
        print("Model ckpt found! Loading...:%s" % model_ckpt_file)
        model.load_weights(model_ckpt_file)

    # Prepare model model saving directory.
    # model_name = "%s.%03d-val_accuracy-{val_accuracy:.4f}.h5" % (MODEL_TYPE, epoch+START_EPOCH)
    model_name = "%s-epoch-{epoch:03d}-auc_good_0-{auc_good_0:.4f}-auc_bad_1-{auc_bad_1:.4f}.h5" % MODEL_TYPE

    filepath = os.path.join(SAVES_DIR, model_name)
    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(
        filepath=filepath, monitor="auc_good_0", verbose=1)
    csv_logger = CSVLogger(".log/training.log.csv", append=True)
    earlystop = EarlyStopping(patience=10)
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    callbacks = [checkpoint, csv_logger,
                 lr_reducer, lr_scheduler]  # 不要 earlystop

    print('Using real-time data augmentation.')
    print("Training Generator...")
    train_datagen = ImageDataGenerator(
        validation_split=0.2,
        rescale=1./255,
        rotation_range=15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        subset='training',
        target_size=IMAGE_SIZE,
        classes=["good_0", "bad_1"],
        color_mode="grayscale",
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42
    )

    print("Class_indices: ", train_generator.class_indices)

    print("Validation Generator...")
    valid_datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
    validation_generator = valid_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        subset='validation',
        target_size=IMAGE_SIZE,
        classes=["good_0", "bad_1"],
        color_mode="grayscale",
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42
    )

    print("Class_indices: ", validation_generator.class_indices)

    print("Fit Model...")
    epochs = 3 if IF_FAST_RUN else TRAINING_EPOCHS
    history = model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=TOTAL_VALIDATE//BATCH_SIZE,
        steps_per_epoch=TOTAL_TRAIN//BATCH_SIZE,
        callbacks=callbacks,
        initial_epoch=START_EPOCH
    )

    print("Save Model...")
    model.save_weights("model-" + MODEL_TYPE + ".h5")

    print("Save History...")
    with open('./history', 'wb') as pickle_file:
        pickle.dump(history.history, pickle_file)


if __name__ == "__main__":
    main()
