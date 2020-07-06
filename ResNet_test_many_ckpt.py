#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-03-20 23:44
# @Author  : Kelly Hwong (you@example.org)
# @Link    : http://example.org

import time
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
from resnet.resnet import model_depth, resnet_v2, lr_schedule

# Parameters we care
exper_name = "ResNet56v2_origin"

test_on_train = True

START_EPOCH = 150  # 已经完成的训练数
ALPHA = 0.99  # label 1 sample's weight
BATCH_SIZE = 32  # 16 for Mac, 64, 128 for server
IF_FAST_RUN = True  # False

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
    BinaryAccuracy(name='accuracy'),  # 整体的 accuracy
    AUC(name='auc_good_0')  # 以 good 为 positive 的 AUC
]

print("If in eager mode: ", tf.executing_eagerly())
print("Use tensorflow version 2.")
assert tf.__version__[0] == "2"

print("Load config ...")
with open('./config/config_win.json', 'r') as f:
    CONFIG = json.load(f)
ROOT_PATH = CONFIG["ROOT_PATH"]
print(f"ROOT_PATH: {ROOT_PATH}")
ROOT_PATH = os.path.expanduser(ROOT_PATH)
print(f"ROOT_PATH: {ROOT_PATH}")
TRAIN_DATA_DIR = os.path.join(ROOT_PATH, CONFIG["TRAIN_DATA_DIR"])
print(f"TRAIN_DATA_DIR: {TRAIN_DATA_DIR}")
TEST_DATA_DIR = os.path.join(ROOT_PATH, CONFIG["TEST_DATA_DIR"])
print(f"TEST_DATA_DIR: {TEST_DATA_DIR}")


print("Prepare testing data...")
if test_on_train:
    num_samples = num_train = 30000
    label_names = os.listdir(TRAIN_DATA_DIR)
    filenames, labels = [], []
    for i, label in enumerate(label_names):
        files = os.listdir(os.path.join(TRAIN_DATA_DIR, label))
        for f in files:
            filenames.append(label+"/"+f)
            labels.append(i)  # 0 or 1
    table = np.asarray([filenames, labels])
    table = table.T
    columns = ["filename", "label"]
    # test on train dataset
    test_df = pd.DataFrame(data=table, columns=columns)
else:
    test_filenames = os.listdir(TEST_DATA_DIR)
    test_df = pd.DataFrame({
        'filename': test_filenames
    })
    num_samples = test_df.shape[0]

print("Create testing generator...")
if test_on_train:
    directory = TRAIN_DATA_DIR
else:
    directory = TEST_DATA_DIR
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df,
    directory,
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    shuffle=False
)  # Found 12500 images.


def create_model(saves_dir, ckpt=None):
    print("Prepare Model")
    n = 6  # order of ResNetv2, 2 or 6
    version = 2
    depth = model_depth(n, version)
    MODEL_TYPE = 'ResNet%dv%d' % (depth, version)
    print(f"MODEL_TYPE: {MODEL_TYPE}")

    model = resnet_v2(input_shape=INPUT_SHAPE, depth=depth, num_classes=2)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=lr_schedule(START_EPOCH)),
                  metrics=METRICS)
    # model.summary()
    if ckpt:
        print("Prepare weights...")
        ckpt_path = os.path.join(saves_dir, ckpt)
        assert os.path.isfile(ckpt_path)
        print(f"Model ckpt {ckpt} found! Loading...:\n%s")
        model.load_weights(ckpt_path)
    return model


MODEL_CKPT = "ResNet56v2-epoch-018-auc-0.9530.h5"

saves_dir = os.path.join(ROOT_PATH, "models-%s/" % exper_name)
if not os.path.exists(saves_dir):
    os.mkdir(saves_dir)

ckpt_files = os.listdir(saves_dir)

for i, ckpt in enumerate(ckpt_files):
    num_ckpt = len(ckpt_files)
    print(f"Test on {i+1}/{num_ckpt} ckpt: {ckpt}.")
    model = create_model(saves_dir, ckpt)

    start = time.perf_counter()
    print("Start testing prediction...")
    pred = model.predict(
        test_generator, steps=np.ceil(num_samples / BATCH_SIZE),
        workers=4, verbose=1)
    elapsed = (time.perf_counter() - start)
    print("Prediction time used:", elapsed)

    if test_on_train:  # test on train
        label = test_df["label"].to_numpy(dtype=int)
        np.save(ckpt + "-pred.npy", pred)
        # score = np.sum(np.abs(-pred[:, 0]))
        # print(f"score: {score}")
    else:  # test on tests
        # predict 第 1 列，是 bad_1 的概率
        test_df['label'] = pred[:, 0]
        print("Predict Samples: ")
        print(type(test_df))
        print(test_df.head(10))
        print("Prepare submission...")
        submission_df = test_df.copy()
        submission_df['id'] = submission_df['filename'].str.split('.').str[0]
        submission_df['label'] = submission_df['label']
        submission_df.drop(['filename', 'label'], axis=1, inplace=True)
        submission_df.to_csv(
            f"./submissions/submission-{exper_name}.csv", index=False)
