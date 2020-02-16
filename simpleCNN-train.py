#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-01-20 06:51
# @Author  : Your Name (you@example.org)
# @Link    : https://www.kaggle.com/uysimty/keras-cnn-dog-or-cat-classification

import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os

import tensorflow as tf
from keras import backend as K
from model import simple_CNN


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


# data path
TRAIN_DATA_DIR = "./data/train/"
SAVES_DIR = "./models-simpleCNN/"


# constants
IF_FAST_RUN = False
EPOCHS_OVER_NIGHT = 50
IMAGE_WIDTH = IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

BATCH_SIZE = 15


def main():
    """ Create Model """
    model_type = "simpleCNN"
    model = simple_CNN(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
    model.summary()
    print(model_type)

    # continuing training
    if os.path.isfile("model-" + model_type + ".h5"):
        print("loading existed model...")
        model.load_weights("model-" + model_type + ".h5")

    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    earlystop = EarlyStopping(patience=10)
    learning_rate_reduction = ReduceLROnPlateau(monitor="auc",
                                                patience=2,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)
    filename = "model_{epoch:02d}-auc-{auc:.4f}.h5"
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(SAVES_DIR, filename), monitor="auc", verbose=1, period=1)
    callbacks = [earlystop, learning_rate_reduction, checkpoint]

    """Prepare Data Frame"""
    filenames = os.listdir(TRAIN_DATA_DIR)
    categories = []
    for filename in filenames:
        category = filename.split('.')[0]
        if category == 'bad':  # bad 1
            categories.append(1)
        else:  # good 0
            categories.append(0)
    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })

    # print(df.head())
    # print(df.tail())
    # df['category'].value_counts().plot.bar()
    # plt.show()

    """Sample Image"""
    # sample = random.choice(filenames)
    # image = load_img("./data/train/"+sample)
    # plt.imshow(image)
    # plt.show()

    """Prepare data"""
    df["category"] = df["category"].replace({0: 'good', 1: 'bad'})

    """ 这里用来自动划分 train 集和 val 集 """
    train_df, validate_df = train_test_split(
        df, test_size=0.20, random_state=42)
    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)

    # train_df['category'].value_counts().plot.bar()

    total_train = train_df.shape[0]
    total_validate = validate_df.shape[0]

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
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    """Validation Generator"""
    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_dataframe(
        validate_df,
        TRAIN_DATA_DIR,
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    """Example Generation"""
    example_df = train_df.sample(n=1).reset_index(drop=True)
    example_generator = train_datagen.flow_from_dataframe(
        example_df,
        TRAIN_DATA_DIR,
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical'
    )

    """Example Generation Ploting"""
    # plt.figure(figsize=(12, 12))
    # for i in range(0, 15):
    #     plt.subplot(5, 3, i+1)
    #     for X_batch, Y_batch in example_generator:
    #         image = X_batch[0]
    #         plt.imshow(image)
    #         break
    # plt.tight_layout()
    # plt.show()

    """Fit Model"""
    epochs = 3 if IF_FAST_RUN else EPOCHS_OVER_NIGHT
    history = model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=total_validate//BATCH_SIZE,
        steps_per_epoch=total_train//BATCH_SIZE,
        callbacks=callbacks
    )

    """Save Model"""
    model.save_weights("model-" + model_type + ".h5")

    """Visualize Training"""
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
