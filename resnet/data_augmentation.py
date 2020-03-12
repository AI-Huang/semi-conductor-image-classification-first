#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Mar-02-20 19:16
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os


def main():
    """ Training Generator """
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
        color_mode="grayscale",
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True
        # seed=42
    )

    """ Validation Generator """
    valid_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    validation_generator = valid_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        subset='validation',
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    """ Example Generation Ploting """
    # plt.figure(figsize=(12, 12))
    # for i in range(0, 15):
    #     plt.subplot(5, 3, i+1)
    #     for X_batch, Y_batch in example_generator:
    #         image = X_batch[0]
    #         plt.imshow(image)
    #         break
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
