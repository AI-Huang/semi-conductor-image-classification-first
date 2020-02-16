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

# constants
IF_FAST_RUN = False
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
    MODEL_CKPT = CONFIG["MODEL_CKPT"]

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
                  #   optimizer=Adam(learning_rate=lr_schedule(0)),
                  optimizer='adam',
                  metrics=['accuracy', auc])
    # model.summary()
    print(MODEL_TYPE)

    # Prepare model model saving directory.
    model_name = "%s.{epoch:03d}-auc-{auc:.4f}.h5" % MODEL_TYPE
    filepath = os.path.join(SAVES_DIR, model_name)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath, monitor="auc", verbose=1)
    earlystop = EarlyStopping(patience=10)
    learning_rate_reduction = ReduceLROnPlateau(monitor="auc",
                                                patience=2,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)
    callbacks = [learning_rate_reduction, checkpoint]  # 不要 earlystop

    """ Resume Training """
    model_ckpt_file = os.path.join(SAVES_DIR, MODEL_CKPT)
    if os.path.exists(model_ckpt_file):
        print("Model ckpt found! Loading...")
        model.load_weights(model_ckpt_file)

    """ Prepare Data Frame """
    filenames = os.listdir(TRAIN_DATA_DIR)
    random.shuffle(filenames)
    categories = []
    for filename in filenames:
        category = filename.split('.')[0]
        if category == "bad":
            categories.append(1)
        else:
            categories.append(0)
    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })
    df["category"] = df["category"].replace({0: "good", 1: "bad"})

    """ 这里用来自动划分 train 集和 val 集 """
    train_df, validate_df = train_test_split(
        df, test_size=0.20, random_state=42)
    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)

    """Traning Generator"""
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        TRAIN_DATA_DIR,
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        class_mode='categorical',
        batch_size=BATCH_SIZE
    )

    # size of train
    total_train = train_df.shape[0]
    total_validate = validate_df.shape[0]

    print('Using real-time data augmentation.')

    """Traning Generator"""
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        TRAIN_DATA_DIR,
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    """ Validation Generator """
    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_dataframe(
        validate_df,
        TRAIN_DATA_DIR,
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    """ Example Generation """
    example_df = train_df.sample(n=1).reset_index(drop=True)
    example_generator = train_datagen.flow_from_dataframe(
        example_df,
        TRAIN_DATA_DIR,
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        class_mode='categorical'
    )

    """ Fit Model """
    epochs = 3 if IF_FAST_RUN else EPOCHS_OVER_NIGHT
    history = model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=total_validate//BATCH_SIZE,
        steps_per_epoch=total_train//BATCH_SIZE,
        callbacks=callbacks
    )

    """ Save Model """
    model.save_weights("model-" + MODEL_TYPE + ".h5")

    """ Save History """
    with open('./history', 'wb') as pickle_file:
        pickle.dump(history.history, pickle_file)


if __name__ == "__main__":
    main()
