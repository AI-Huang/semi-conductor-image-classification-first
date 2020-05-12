#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Mar-02-20 19:16
# @Author  : Your Name (you@example.org)
# @Link    : https://www.kaggle.com/uysimty/keras-cnn-dog-or-cat-classification

import os
import json
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

IMAGE_WIDTH = IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)


def main():
    """ Load Config """
    with open('./config/config_origin.json', 'r') as f:
        CONFIG = json.load(f)
    BATCH_SIZE = CONFIG["BATCH_SIZE"]
    ROOT_PATH = CONFIG["ROOT_PATH"]
    TRAIN_DATA_DIR = CONFIG["TRAIN_DATA_DIR"]
    SAMPLE_DATA_DIR = CONFIG["SAMPLE_DATA_DIR"]
    TRAIN_DATA_DIR = os.path.join(ROOT_PATH, TRAIN_DATA_DIR)
    SAMPLE_DATA_DIR = os.path.join(ROOT_PATH, SAMPLE_DATA_DIR)
    MODEL_CKPT = CONFIG["MODEL_CKPT"]

    """ Training Generator """
    print('Using real-time data augmentation.')
    train_datagen = ImageDataGenerator(
        validation_split=0.2,
        rescale=1./255,
        rotation_range=15,  # augmentation 的个数
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
        color_mode="grayscale",
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42
    )

    """ Validation Generator """
    valid_datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
    validation_generator = valid_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        subset='validation',
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42
    )

    # example_df = train_df.sample(n=1).reset_index(drop=True)
    example_generator = train_datagen.flow_from_directory(
        SAMPLE_DATA_DIR,
        subset='training',
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42
    )

    """ Example Batch Ploting """
    # plt.figure(figsize=(12, 12))
    # plt.title("sample images, [1, 0] for bad, [0, 1] for good")
    # for X_batch, Y_batch in example_generator:
    #     for i in range(0, BATCH_SIZE):
    #         plt.subplot(4, 4, i+1)
    #         image = X_batch[i]
    #         label = Y_batch[i]
    #         label_str = ','.join(str(_) for _ in label)
    #         print("label:", label)
    #         print(image.shape)
    #         image = image.squeeze()
    #         print(image.shape)
    #         plt.title("label:" + label_str)
    #         plt.imshow(image)
    #     break  # only show one batch
    # plt.tight_layout()
    # plt.show()

    """ Example Augmentation Ploting """
    plt.figure(figsize=(12, 12))
    for i in range(0, 15):
        plt.subplot(3, 5, i+1)
        for X_batch, Y_batch in example_generator:  # generator 按批生成图片，再按批生成它们的 augmentation
            image = X_batch[0]
            label = Y_batch[0]
            label_str = ','.join(str(_) for _ in label)
            print("label:", label)
            image = image.squeeze()
            plt.title("label:" + label_str)
            plt.imshow(image)
            break
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
