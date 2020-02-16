#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-11-20 14:18
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Link    : https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# data path
MODEL_SAVES_DIR = "./models-resnetv2/"
MODEL_FILE_NAME = "ResNet20v2.025-auc-0.9367.h5"


def list_model():
    model_list = os.listdir(MODEL_SAVES_DIR)
    model_list = [_.replace("ResNet20v2.", "").replace(".h5", "")
                  for _ in model_list]
    nums = [int(_.split('-')[0]) for _ in model_list]
    base_num = 25  # 在此之前训练了这么多epoch
    aucs = [float(_.split('-')[2]) for _ in model_list]
    print(nums)
    print(aucs)
    with open("./epoch-auc.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "auc"])
        for i in range(len(nums)):
            writer.writerow([nums[i], aucs[i]])


def plot_auc():
    base_num = 25  # 在此之前训练了这么多epoch
    epoch_auc = []
    with open("./epoch-auc.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            print(row)
            epoch_auc.append([int(row[0])+base_num, float(row[1])])
    epoch_auc = np.asarray(epoch_auc)
    print(epoch_auc)
    epoch_auc = epoch_auc[epoch_auc[:, 0].argsort()]  # 按照第一列排序
    print(epoch_auc)
    """ Visualize Training """
    plt.plot(epoch_auc[:, 0], epoch_auc[:, 1], color='r', label="Training auc")
    # plt.set_xticks(np.arange(1, epochs, 1))
    # plt.set_yticks(np.arange(0, 1, 0.1))
    plt.xlabel("num_epoch")
    plt.ylabel("train_auc")
    legend = plt.legend(loc='best', shadow=True)
    plt.tight_layout()
    plt.show()


def main():
    plot_auc()


if __name__ == "__main__":
    main()
