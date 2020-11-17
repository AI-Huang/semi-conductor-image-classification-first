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
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.metrics import Recall, Precision, TruePositives, FalsePositives, TrueNegatives, FalseNegatives, BinaryAccuracy, AUC
from tensorflow.keras.losses import BinaryCrossentropy
from resnet import model_depth, resnet_v2, lr_schedule

# Parameters we care
START_EPOCH = 0  # 已经完成的训练数
ALPHA, GAMMA = 0.99, 1  # label 1 sample's weight
BATCH_SIZE = 16  # 16 for Mac, 64, 128 for server
IF_FAST_RUN = False  # False
print(f"ALPHA: {ALPHA}")

# Training parameters
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
    Recall(name='recall'),
    Precision(name='precision'),
    TruePositives(name='tp'),  # thresholds=0.5
    FalsePositives(name='fp'),
    TrueNegatives(name='tn'),
    FalseNegatives(name='fn'),
    BinaryAccuracy(name='accuracy'),
    #     AUC0(name='auc_good_0'),  # 以 good 为 positive 的 AUC
    AUC(name='auc_bad_1')  # 以 bad 为 positive 的 AUC
    #     AUC(name='auc_good_0')  # 以 good 为 positive 的 AUC
]

print("If in eager mode: ", tf.executing_eagerly())
print(f"Use tensorflow version: {tf.__version__}.")
assert tf.__version__[0] == "2"

print("Load Config ...")
with open('./config.json', 'r') as f:  # by default
    CONFIG = json.load(f)
ROOT_PATH = CONFIG["ROOT_PATH"]
print(f"ROOT_PATH: {ROOT_PATH}")
ROOT_PATH = os.path.expanduser(ROOT_PATH)
print(f"ROOT_PATH: {ROOT_PATH}")
TRAIN_DATA_DIR = os.path.join(ROOT_PATH, CONFIG["TRAIN_DATA_DIR"])
print(f"TRAIN_DATA_DIR: {TRAIN_DATA_DIR}")

print("Prepare Model")
n = 6  # order of ResNetv2, 2 or 6
version = 2
depth = model_depth(n, version)
MODEL_TYPE = 'ResNet%dv%d' % (depth, version)
SAVES_DIR = os.path.join(ROOT_PATH, "models-%s/" % MODEL_TYPE)
if not os.path.exists(SAVES_DIR):
    os.mkdir(SAVES_DIR)
MODEL_CKPT = os.path.join(
    SAVES_DIR, "ResNet56v2-epoch-149-auc_good_0-0.9882-auc_bad_1-0.9886.h5")  # CONFIG["MODEL_CKPT"]
print(f"MODEL_CKPT: {MODEL_CKPT}")

model = resnet_v2(input_shape=INPUT_SHAPE, depth=depth, num_classes=2)
model.compile(loss=BinaryCrossentropy(),
              optimizer=Adam(learning_rate=lr_schedule(epoch=0)),
              metrics=METRICS)
# model.summary()
print(MODEL_TYPE)

# print("Resume Training...")
# model_ckpt_file = MODEL_CKPT
# assert os.path.isfile(model_ckpt_file)
# print("Model ckpt found! Loading...:%s" % model_ckpt_file)
# model.load_weights(model_ckpt_file)

# Model model saving directory.
# ckpt_name = "%s-epoch-{epoch:03d}-auc_good_0-{auc_good_0:.4f}-auc_bad_1-{auc_bad_1:.4f}.h5" % MODEL_TYPE
ckpt_name = "%s-epoch-{epoch:03d}-auc_bad_1-{auc_bad_1:.4f}.h5" % MODEL_TYPE

filepath = os.path.join(SAVES_DIR, ckpt_name)
# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(
    filepath=filepath, monitor="auc_bad_1", verbose=1)
csv_logger = CSVLogger("./log/training.log.csv", append=True)
earlystop = EarlyStopping(patience=10)
lr_scheduler = LearningRateScheduler(
    lr_schedule, verbose=1)  # verbose>0, 打印 lr_scheduler 的信息
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
callbacks = [checkpoint, csv_logger,
             lr_reducer, lr_scheduler]  # 不要 earlystop

classes = ["good_0", "bad_1"]
# classes=["bad_1", "good_0"] # Previous mapping

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
    batch_size=BATCH_SIZE,
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
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

print("Train class_indices: ", train_generator.class_indices)
print("Val class_indices: ", validation_generator.class_indices)

print("Fit Model...")

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_visible_devices(gpus[3], 'GPU')

epochs = 3 if IF_FAST_RUN else TRAINING_EPOCHS

# class_weight = {0: 1.0/101, 1: 100.0/101} # class_weight
# print("class_weight:", class_weight)

with tf.device('/device:GPU:2'):
    history = model.fit(
        train_generator,
        epochs=epochs,
        #         class_weight=class_weight,  # 添加 class_weight
        validation_data=validation_generator,
        validation_steps=TOTAL_VALIDATE//BATCH_SIZE,
        steps_per_epoch=TOTAL_TRAIN//BATCH_SIZE,
        callbacks=callbacks,
        initial_epoch=START_EPOCH
    )
