#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-28-20 22:56
# @Update  : Nov-23-20 20:57
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @Link    : http://example.org


import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

from utils import read_data
from utils import read_TP, read_TN

# data path
MODEL_SAVES_DIR = "./models-resnetv2/"
MODEL_FILE_NAME = "ResNet20v2.025-auc-0.9367.h5"

# constants
IMAGE_WIDTH = IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
BATCH_SIZE = 16

CURRENT_BEST = 0.98866
MY_BEST = 0.97156


def plot_scores(epoch_auc):
    """ Load Config """
    with open('config.json', 'r') as f:
        CONFIG = json.load(f)
    current_best = CONFIG["CURRENT_BEST"]
    my_best = CONFIG["MY_BEST"]

    """ Visualize Training """
    plt.figure(figsize=(1000/100, 800/100))
    plt.plot(epoch_auc[:, 0], epoch_auc[:, 1],
             color='r', linestyle='-', marker='*', label="Training auc")
    plt.plot(epoch_auc[:, 0], epoch_auc[:, 2],
             color='b', linestyle='-', marker='o', label="Score")
    # 随便选两个epoch坐标把best画出来
    plt.scatter(45, current_best, label="Current best")
    plt.scatter(40, my_best, label="My best")

    # plt.set_xticks(np.arange(1, epochs, 1))
    # plt.set_yticks(np.arange(0, 1, 0.1))
    # 设置坐标轴范围
    plt.xlim((20, 50))
    plt.ylim((0.9, 1))
    plt.xlabel("num_epoch")
    plt.ylabel("train_auc")
    # 设置坐标轴刻度
    plt.xticks(np.arange(20, 50, 1))
    plt.yticks(np.arange(0.9, 1, 0.01))

    plt.grid(True)
    legend = plt.legend(loc='best', shadow=True)
    plt.tight_layout()
    plt.show()


def plot_scores2():
    """ Load Config """
    with open('config.json', 'r') as f:
        CONFIG = json.load(f)
    current_best = CURRENT_BEST
    my_best = MY_BEST

    """ Read data """
    data_acc = read_data(filepath="./ResNet20v2-epoch-auc.csv")
    epochs_resnet20 = 45
    epoch_val_accuracy = read_data(
        filepath="./ResNet56v2-epoch-val_accuracy.csv")
    epoch_auc = read_data(
        filepath="./ResNet56v2-epoch-auc.csv")
    for i in range(len(epoch_val_accuracy)):
        epoch_val_accuracy[i, 0] += epochs_resnet20
    for i in range(len(epoch_auc)):
        epoch_auc[i, 0] += epochs_resnet20
    """ Visualize Training """
    plt.figure(figsize=(1200/100, 800/100))
    plt.plot(data_acc[:, 0], data_acc[:, 1],
             color='r', linestyle='-', marker='*', label="Training auc")
    plt.plot(data_acc[:, 0], data_acc[:, 2],
             color='b', linestyle='-', marker='o', label="Score")
    # 随便选两个epoch坐标把best画出来
    plt.scatter(45.5, my_best, color='g', label="My best")
    plt.scatter(60, current_best, color='m', label="Current best")

    plt.plot(epoch_val_accuracy[:, 0], epoch_val_accuracy[:, 1],
             color='r', linestyle='-', marker='*', label="Training accuracy")
    plt.plot(epoch_val_accuracy[:, 0], epoch_val_accuracy[:, 2],
             color='b', linestyle='-', marker='o', label="Score")
    plt.plot(epoch_auc[:, 0], epoch_auc[:, 1],
             color='r', linestyle='-', marker='*', label="Training auc")
    plt.plot(epoch_auc[:, 0], epoch_auc[:, 2],
             color='b', linestyle='-', marker='o', label="Score")

    # plt.set_xticks(np.arange(1, epochs, 1))
    # plt.set_yticks(np.arange(0, 1, 0.1))
    # 设置坐标轴范围
    EPOCHS = 70
    plt.xlim((20, EPOCHS))
    plt.ylim((0.65, 1))
    plt.xlabel("num_epoch")
    plt.ylabel("train_auc/acc/score")
    # 设置坐标轴刻度
    plt.xticks(np.arange(20, EPOCHS, 1))
    plt.yticks(np.arange(0.65, 1, 0.01))
    plt.grid(True)
    legend = plt.legend(loc='best', shadow=True)
    plt.tight_layout()
    plt.show()


def plot_train_samples():
    from keras.preprocessing.image import ImageDataGenerator, load_img
    """ Load Config """
    with open('config.json', 'r') as f:
        CONFIG = json.load(f)
    ROOT_PATH = CONFIG["ROOT_PATH"]
    TRAIN_DATA_DIR = CONFIG["TRAIN_DATA_DIR"]
    TRAIN_DATA_DIR = os.path.join(ROOT_PATH, TRAIN_DATA_DIR)

    """ Data """
    print(TRAIN_DATA_DIR)
    bad_datagen = ImageDataGenerator(rescale=1./255)
    bad_sample_generator = bad_datagen.flow_from_directory(
        os.path.join(TRAIN_DATA_DIR, "train_bad"),
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    good_datagen = ImageDataGenerator(rescale=1./255)
    good_sample_generator = good_datagen.flow_from_directory(
        os.path.join(TRAIN_DATA_DIR, "train_good"),
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    """ Plot """
    plt.figure(figsize=(12, 12))
    # axes = plt.subplot(1, 2)
    for X_batch, Y_batch in bad_sample_generator:
        for i in range(BATCH_SIZE):
            plt.subplot(4, 4, i+1)
            image = X_batch[i]
            image = image[:, :, 0]
            plt.imshow(image, cmap='gray')
            # plt.imshow(image) # RGB share same values
        break  # only one batch
    plt.tight_layout()
    plt.show()

    for X_batch, Y_batch in good_sample_generator:
        for i in range(BATCH_SIZE):
            plt.subplot(4, 4, i+1)
            image = X_batch[i]
            image = image[:, :, 0]
            plt.imshow(image, cmap='gray')
            # plt.imshow(image) # RGB share same values
        break  # only one batch
    plt.tight_layout()
    plt.show()


def plot_simpleCNN():
    """ Read data """
    data_acc = read_data(filepath="./simpleCNN-epoch-acc.csv")
    data_auc = read_data(filepath="./simpleCNN-epoch-auc.csv")
    # print(data_acc)
    """ Visualize Training """
    plt.plot(data_acc[:, 0], data_acc[:, 1],
             color='r', linestyle='-', marker='v', label="Training accuracy")
    plt.plot(data_acc[:, 0], data_acc[:, 2],
             color='b', linestyle='-', marker='o', label="Score")
    plt.text(data_acc[-1, 0], data_acc[-1, 2], (data_acc[-1, 0],
                                                data_acc[-1, 2]), ha='center', va='bottom', fontsize=10)
    plt.plot(data_auc[:, 0], data_auc[:, 1],
             color='r', linestyle='-', marker='^', label="Training AUC")
    plt.plot(data_auc[:, 0], data_auc[:, 2],
             color='b', linestyle='-', marker='o', label="Score")
    plt.text(data_auc[-1, 0], data_auc[-1, 2], (data_auc[-1, 0],
                                                data_auc[-1, 2]), ha='center', va='bottom', fontsize=10)
    EPOCHS = 45
    plt.xlim((0, EPOCHS))
    plt.ylim((0, 1))
    plt.xlabel("num_epoch")
    plt.ylabel("train_auc/acc/score")
    # 设置坐标轴刻度
    plt.xticks(np.arange(0, EPOCHS, 1))
    plt.yticks(np.arange(0, 1, 0.1))
    plt.grid(True)
    legend = plt.legend(loc='best', shadow=True)
    plt.show()


def plot_ResNet20():
    """ Read data """
    data_auc = read_data(filepath="./ResNet20v2-epoch-auc.csv")
    """ Visualize Training """
    plt.plot(data_auc[:, 0], data_auc[:, 1],
             color='r', linestyle='-', marker='^', label="ResNet20 Training AUC")
    plt.plot(data_auc[:, 0], data_auc[:, 2],
             color='b', linestyle='-', marker='o', label="ResNet20 Score")
    for e in data_auc:
        if e[2] > 0:
            plt.text(e[0], e[2], (e[0],
                                  e[2]), ha='center', va='bottom', fontsize=10)
    EPOCHS = 50
    plt.xlim((0, EPOCHS))
    plt.ylim((0, 1))
    plt.xlabel("num_epoch")
    plt.ylabel("train_auc/acc/score")
    # 设置坐标轴刻度
    plt.xticks(np.arange(0, EPOCHS, 5))
    plt.yticks(np.arange(0, 1, 0.1))
    plt.grid(True)
    legend = plt.legend(loc='best', shadow=True)
    plt.show()


def plot_ResNet56():
    """ Read data """
    data_acc = read_data(filepath="./ResNet56v2-epoch-val_accuracy.csv")
    data_auc = read_data(filepath="./ResNet56v2-epoch-auc.csv")
    # print(data_acc)
    """ Visualize Training """
    plt.plot(data_acc[:, 0], data_acc[:, 1],
             color='r', linestyle='-', marker='v', label="Training accuracy")
    plt.plot(data_acc[:, 0], data_acc[:, 2],
             color='b', linestyle='-', marker='o', label="Score")
    for e in data_acc:
        if e[2] > 0:
            plt.text(e[0], e[2], (e[0],
                                  e[2]), ha='center', va='bottom', fontsize=10)
    plt.plot(data_auc[1:, 0], data_auc[1:, 1],  # epoch 0 不要
             color='r', linestyle='-', marker='^', label="Training AUC")
    plt.plot(data_auc[1:, 0], data_auc[1:, 2],
             color='b', linestyle='-', marker='o', label="Score")
    for e in data_auc:
        if e[2] > 0:
            plt.text(e[0], e[2], (e[0],
                                  e[2]), ha='center', va='bottom', fontsize=10)
    EPOCHS = 110
    plt.xlim((0, EPOCHS))
    plt.ylim((0, 1))
    plt.xlabel("num_epoch")
    plt.ylabel("train_auc/acc/score")
    # 设置坐标轴刻度
    plt.xticks(np.arange(0, EPOCHS, 5))
    plt.yticks(np.arange(0, 1, 0.1))
    plt.grid(True)
    legend = plt.legend(loc='best', shadow=True)
    plt.show()


def plot_TPR():
    thresholds = np.linspace(0, 1, num=200)
    # print(thresholds)
    # threshold = 0.5
    tprs = []
    fprs = []
    for threshold in thresholds:
        tp, fn = read_TP(
            "./TPR/TP-ResNet56v2.109-auc-0.9832.h5.csv", threshold=threshold)
        tn, fp = read_TN(
            "./TPR/TN-ResNet56v2.109-auc-0.9832.h5.csv", threshold=threshold)
        # print(tp, fn)  # 444 156, of 600
        # print(tn, fp)  # 5357 43, of 5400
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        tprs.append(tpr)
        fprs.append(fpr)
        # print(tpr, fpr)
    plt.plot(fprs, tprs)
    return None


def main():
    # plot_simpleCNN()
    # plot_ResNet56()
    # plot_ResNet20()
    plot_TPR()
    plt.legend()
    plt.show()
    return
    # plot_train_samples()

    # 随便选两个epoch坐标把best画出来
    plt.scatter(45.5, MY_BEST, color='g', label="My best")
    plt.scatter(60, CURRENT_BEST, color='m', label="Current best")

    # unify_epoch()
    """ Load Config """
    with open('config.json', 'r') as f:
        CONFIG = json.load(f)
    current_best = CONFIG["CURRENT_BEST"]
    my_best = CONFIG["MY_BEST"]

    """ Read data """
    epoch_auc_resnet56 = read_data(filepath="./ResNet56v2-epoch-auc.csv")
    """ Visualize Training """
    plt.figure(figsize=(1200/100, 800/100))
    # 随便选两个epoch坐标把best画出来
    # plt.scatter(45.5, my_best, color='g', label="My best")
    # plt.scatter(60, current_best, color='m', label="Current best")

    # TODO subplot
    plt.plot(epoch_auc_resnet56[:, 0], epoch_auc_resnet56[:, 1],
             color='r', linestyle='-', marker='*', label="Training accuracy")
    plt.plot(epoch_auc_resnet56[:, 0], epoch_auc_resnet56[:, 2],
             color='b', linestyle='-', marker='o', label="Score")

    # 设置坐标轴刻度
    plt.xticks(np.arange(0, 120, 5))
    plt.yticks(np.arange(0, 1, 0.01))
    plt.grid(True)
    legend = plt.legend(loc='best', shadow=True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
