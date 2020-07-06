#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-03-20 23:44
# @Author  : Kelly Hwong (you@example.org)
# @Link    : http://example.org

import os
import sys
import json
import random
import pickle
import numpy as np
import pandas as pd
from optparse import OptionParser

import tensorflow as tf  # to check backend
from tensorflow import keras  # if we want tf2 Keras, not standalone Keras
# import keras
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.metrics import Recall, Precision, TruePositives, FalsePositives, TrueNegatives, FalseNegatives, BinaryAccuracy, AUC
from resnet.resnet import model_depth, resnet_v2, lr_schedule

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
    Recall(name='recall'),
    Precision(name='precision'),
    TruePositives(name='tp'),  # thresholds=0.5
    FalsePositives(name='fp'),
    TrueNegatives(name='tn'),
    FalseNegatives(name='fn'),
    BinaryAccuracy(name='accuracy'),
    AUC(name='auc')  # positive class 下的 AUC
]


def cmd_parser():
    parser = OptionParser()
    # Parameters we care
    parser.add_option('--start_epoch', type='int', dest='start_epoch',
                      action='store', default=0, help='start_epoch, i.e., epoches that have been trained, e.g. 80.')  # 已经完成的训练数
    parser.add_option('--batch_size', type='int', dest='batch_size',
                      action='store', default=16, help='batch_size, e.g. 16.')  # 16 for Mac, 64, 128 for server
    parser.add_option('--train_epochs', type='int', dest='train_epochs',
                      action='store', default=150, help='train_epochs, e.g. 150.')  # training 150 epochs to fit enough
    # parser.add_option('--if_fast_run', type='choice', dest='if_fast_run',
    #   action='store', default=0.99, help='') # TODO
    parser.add_option('--loss', type='string', dest='loss',
                      action='store', default="bce", help='loss name, e.g., bce or cce.')
    parser.add_option('--exper_name', type='string', dest='exper_name',
                      action='store', default="ResNet56v2", help='exper_name, user named experiment name, e.g., ResNet56v2_BCE.')
    parser.add_option('--config_file', type='string', dest='config_file',
                      action='store', default="./config/config.json", help='config_file path, e.g., ./config/config.json.')
    parser.add_option('--ckpt', type='string', dest='ckpt',
                      action='store', default="", help='ckpt, model ckpt file.')
    parser.add_option('--positive_class', type='string', dest='positive_class',
                      action='store', default="bad_1", help='ckpt, model ckpt file, e.g. bad_1 or good_0.')
    parser.add_option('--alpha', type='float', dest='alpha',
                      action='store', default=0.99, help='alpha for focal loss if this loss is used.')
    parser.add_option('--n', type='int', dest='n',
                      action='store', default=6, help='n, order of ResNet, 2 or 6.')
    parser.add_option('--version', type='int', dest='version',
                      action='store', default=2, help='version, version of ResNet, 1 or 2.')

    args, _ = parser.parse_args(sys.argv[1:])
    return args


def main():
    options = cmd_parser()
    loss = options.loss
    print(f"loss: {loss}.")
    if loss == "bce":
        from tensorflow.keras.losses import BinaryCrossentropy
        loss = BinaryCrossentropy()
    elif loss == "cce":
        from tensorflow.keras.losses import CategoricalCrossentropy
        loss = CategoricalCrossentropy()
    else:
        print("您输入的loss有误，将使用默认的 BCE loss。")
        from tensorflow.keras.losses import BinaryCrossentropy
        loss = BinaryCrossentropy()
    print(loss)

    classes = ["good_0", "bad_1"]
    if options.positive_class == "good_0":
        classes = ["bad_1", "good_0"]

    if_fast_run = False
    print(f"TensorFlow version: {tf.__version__}.")  # Keras backend
    print(f"Keras version: {keras.__version__}.")
    print("If in eager mode: ", tf.executing_eagerly())
    assert tf.__version__[0] == "2"

    print("Load Config ...")
    with open(options.config_file, 'r') as f:
        CONFIG = json.load(f)
    ROOT_PATH = CONFIG["ROOT_PATH"]
    print(f"ROOT_PATH: {ROOT_PATH}")
    ROOT_PATH = os.path.expanduser(ROOT_PATH)
    print(f"ROOT_PATH: {ROOT_PATH}")
    TRAIN_DATA_DIR = os.path.join(ROOT_PATH, CONFIG["TRAIN_DATA_DIR"])
    print(f"TRAIN_DATA_DIR: {TRAIN_DATA_DIR}")

    print("Prepare Model")
    n = options.n
    version = options.version
    depth = model_depth(n, version)
    MODEL_TYPE = 'ResNet%dv%d' % (depth, version)
    SAVES_DIR = "models-%s/" % options.exper_name
    SAVES_DIR = os.path.join(ROOT_PATH, SAVES_DIR)
    if not os.path.exists(SAVES_DIR):
        os.mkdir(SAVES_DIR)

    model = resnet_v2(input_shape=INPUT_SHAPE, depth=depth, num_classes=2)
    model.compile(loss=loss,
                  optimizer=Adam(learning_rate=lr_schedule(
                      options.start_epoch)),
                  metrics=METRICS)
    # model.summary()
    print(MODEL_TYPE)

    model_ckpt_file = options.ckpt
    if model_ckpt_file != "" and os.path.exists(os.path.join(SAVES_DIR, model_ckpt_file)):
        model_ckpt_file = os.path.join(SAVES_DIR, model_ckpt_file)
        print("Model ckpt found! Loading...:%s" % model_ckpt_file)
        print("Resume Training...")
        model.load_weights(model_ckpt_file)

    # Prepare callbacks for model saving and for learning rate adjustment.
    model_name = "%s-epoch-{epoch:03d}-auc-{auc:.4f}.h5" % MODEL_TYPE
    filepath = os.path.join(SAVES_DIR, model_name)
    checkpoint = ModelCheckpoint(
        filepath=filepath, monitor="auc", verbose=1)
    csv_logger = CSVLogger(
        f"./log/training.log.{options.exper_name}.csv", append=True)
    earlystop = EarlyStopping(patience=10)
    lr_scheduler = LearningRateScheduler(
        lr_schedule, verbose=1)  # verbose>0, 打印 lr_scheduler 的信息
    lr_reducer = ReduceLROnPlateau(monitor="auc",
                                   patience=2,
                                   verbose=1,
                                   factor=0.5,
                                   min_lr=0.00001)
    callbacks = [checkpoint, csv_logger,
                 lr_reducer]  # 不要 earlystop, lr_scheduler

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
        classes=classes,
        color_mode="grayscale",
        class_mode='categorical',
        batch_size=options.batch_size,
        shuffle=True,
        seed=42
    )

    print("Validation Generator...")
    valid_datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
    validation_generator = valid_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        subset='validation',
        target_size=IMAGE_SIZE,
        classes=classes,
        color_mode="grayscale",
        class_mode='categorical',
        batch_size=options.batch_size,
        shuffle=True,
        seed=42
    )

    print("Train class_indices: ", train_generator.class_indices)
    print("Val class_indices: ", validation_generator.class_indices)

    print("Fit Model...")
    epochs = 3 if if_fast_run else options.train_epochs
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=TOTAL_VALIDATE//options.batch_size,
        steps_per_epoch=TOTAL_TRAIN//options.batch_size,
        callbacks=callbacks,
        initial_epoch=options.start_epoch
    )


if __name__ == "__main__":
    main()
