#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-01-20 06:51
# @Author  : Your Name (you@example.org)
# @Link    : https://www.kaggle.com/uysimty/keras-cnn-dog-or-cat-classification

import os
import json
import random
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import backend as K
from model import simple_CNN_auc, auc

# Training parameters
START_EPOCH = 20
IF_FAST_RUN = False
EPOCHS_OVER_NIGHT = 50
# BATCH_SIZE = 15  # 16 # orig paper trained all networks with batch_size=128
TOTAL_TRAIN = 30000 * 0.8
TOTAL_VALIDATE = 30000 * 0.2

# constants
IF_DATA_AUGMENTATION = True
NUM_CLASSES = 2
IMAGE_WIDTH = IMAGE_HEIGHT = 128  # CNN 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3
INPUT_SHAPE = [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS]


def main():
    """ Load Config """
    with open('config.json', 'r') as f:
        CONFIG = json.load(f)
    BATCH_SIZE = CONFIG["BATCH_SIZE"]
    ROOT_PATH = CONFIG["ROOT_PATH"]
    TRAIN_DATA_DIR = CONFIG["TRAIN_DATA_DIR"]
    TEST_DATA_DIR = CONFIG["TEST_DATA_DIR"]
    TRAIN_DATA_DIR = os.path.join(ROOT_PATH, TRAIN_DATA_DIR)
    TEST_DATA_DIR = os.path.join(ROOT_PATH, TEST_DATA_DIR)

    MODEL_CKPT = CONFIG["MODEL_CKPT"]

    # Create model
    MODEL_TYPE = "simpleCNN"
    SAVES_DIR = "models-%s/" % MODEL_TYPE
    SAVES_DIR = os.path.join(ROOT_PATH, SAVES_DIR)

    if not os.path.exists(SAVES_DIR):
        os.mkdir(SAVES_DIR)
    model = simple_CNN_v2_auc(INPUT_SHAPE)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy', auc])
    model.summary()
    print(MODEL_TYPE)

    # Resume training
    if os.path.isfile("model-" + MODEL_TYPE + ".h5"):
        print("loading existed model...")
        model.load_weights("model-" + MODEL_TYPE + ".h5")

    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    filename = "model_{epoch:02d}-auc-{auc:.4f}.h5"
    filepath = os.path.join(SAVES_DIR, filename)
    checkpoint = ModelCheckpoint(
        filepath=filepath, monitor="auc", verbose=1, period=1)
    earlystop = EarlyStopping(patience=10)
    learning_rate_reduction = ReduceLROnPlateau(monitor="auc",
                                                patience=2,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)
    callbacks = [learning_rate_reduction, checkpoint]  # 不要 earlystop

    # Training generator
    print('Using real-time data augmentation.')
    train_datagen = ImageDataGenerator(
        validation_split=0.2,
        rotation_range=15,
        rescale=1./255,
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
        # color_mode="grayscale",
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True
        # seed=42
    )

    # Validation generator
    valid_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    validation_generator = valid_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        subset='validation',
        target_size=IMAGE_SIZE,
        # color_mode="grayscale",
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Fit model
    epochs = 3 if IF_FAST_RUN else EPOCHS_OVER_NIGHT
    history = model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=TOTAL_VALIDATE//BATCH_SIZE,
        steps_per_epoch=TOTAL_TRAIN//BATCH_SIZE,
        callbacks=callbacks
    )

    # Save model
    model.save_weights("model-" + MODEL_TYPE + ".h5")

    # Visualize training
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    ax1.plot(history.history['loss'], color='b', label="Training loss")
    ax1.plot(history.history['val_loss'], color='r', label="validation loss")
    ax1.set_xticks(np.arange(1, epochs, 1))
    ax1.set_yticks(np.arange(0, 1, 0.1))

    ax2.plot(history.history['acc'], color='b', label="Training accuracy")
    ax2.plot(history.history['val_acc'], color='r',
             label="Validation accuracy")
    ax2.set_xticks(np.arange(1, epochs, 1))

    legend = plt.legend(loc='best', shadow=True)
    plt.tight_layout()

    # TODO plot.save
    plt.show()


if __name__ == "__main__":
    main()
