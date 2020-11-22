#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Nov-19-20 14:05
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @Link    : http://example.org

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data loader for semi-conductor dataset


def data_generators(data_dir, target_size, batch_size=32, classes=["good_0", "bad_1"], seed=42):
    """data_generators
    Inputs:
        data_dir:
        target_size:
        batch_size: train generator and validation generator's batch_size, default 32.
        classes: default '["good_0", "bad_1"]'
        seed: train generator's random seed, default 42.
    Return:
        train_generator:
        validation_generator:
    """

    # total_train = train_df.shape[0]
    # total_validate = validate_df.shape[0]

    # Training Generator
    # Using real-time data augmentation
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
        os.path.join(data_dir, "train"),
        subset='training',
        target_size=target_size,
        color_mode="grayscale",
        classes=classes,
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True,
        seed=seed
    )

    # Validation Generator
    valid_datagen = ImageDataGenerator(validation_split=0.2)
    validation_generator = valid_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        subset='validation',
        target_size=target_size,
        color_mode="grayscale",
        classes=classes,
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False
    )

    return train_generator, validation_generator


def get_test_generator(data_dir, target_size, batch_size):
    """get_test_generator
    Inputs:
        data_dir:
        target_size:
        batch_size:
    Return:
        test_generator:
        test_df:
    """
    # Prepare DataFrame
    test_filenames = os.listdir(os.path.join(data_dir, "test"))
    test_df = pd.DataFrame({
        'filename': test_filenames
    })
    # num_samples = test_df.shape[0]

    # Test Generator
    test_gen = ImageDataGenerator(rescale=1./255)
    test_generator = test_gen.flow_from_dataframe(
        test_df,
        directory=os.path.join(data_dir, "test"),
        x_col='filename',
        y_col=None,
        class_mode=None,
        target_size=target_size,
        color_mode="rgb",
        batch_size=batch_size,
        shuffle=False
    )  # Found 12500 images.

    return test_generator, test_df
