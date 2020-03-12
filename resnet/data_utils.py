#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-11-20 14:18
# @Author  : Kelly Hwong (you@example.org)
# @Link    : https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column

import os
import csv
import json
import numpy as np
import shutil
from tqdm import tqdm


def change_positive_column():
    with open("submission.csv", "r") as submission_origin:
        f_origin = csv.reader(submission_origin)
        headers = next(f_origin)
        new_rows = []
        for row in f_origin:
            # print(row)
            new_score = str(1 - float(row[1]))
            new_row = [row[0], new_score]
            new_rows.append(new_row)

    with open('submission_modified.csv', 'w') as submission_modified:
        f_modified = csv.writer(submission_modified)
        f_modified.writerow(headers)
        for row in new_rows:
            f_modified.writerow(row)


def read_data(filepath=""):
    data = []
    with open(filepath, "r") as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            # print("row:", row)
            if row[2] == "":
                row[2] = 0  # score, 没有就补0
            data.append([int(row[0]), float(row[1]), float(row[2])])
    data = np.asarray(data)
    data = data[data[:, 0].argsort()]  # 按照第一列排序
    return data


def list_model_epochs(model_dir, model_type):
    _model_list = os.listdir(model_dir)
    model_list = []
    for m in _model_list:
        if "auc" in m:
            model_list.append(m.replace(model_type+".", "").replace(".h5", ""))
    epochs = [int(_.split('-')[0]) for _ in model_list]
    aucs = [float(_.split('-')[2]) for _ in model_list]
    return epochs, aucs


def unify_epoch():
    """ Change "ResNet56v2.85and001-auc-0.9811" to "ResNet56v2.086-auc-0.9811"
    Only run once!
    """
    """ Load Config """
    with open('config.json', 'r') as f:
        CONFIG = json.load(f)
    ROOT_PATH = CONFIG["ROOT_PATH"]
    MODEL_DIR = CONFIG["MODEL_DIR"]
    MODEL_DIR = os.path.join(ROOT_PATH, MODEL_DIR)

    os.chdir(MODEL_DIR)
    for filename in os.listdir(MODEL_DIR):
        if "and" in filename:
            # print(filename)
            i = filename.find("and")
            # print(filename[i-2:i], filename[i+3:i+6])
            start_epoch = int(filename[i-2:i])
            delta_epoch = int(filename[i+3:i+6])
            epoch = start_epoch + delta_epoch
            new_filename = filename[:i-2] + "%03d" % epoch + filename[i+6:]
            print(new_filename)
            input("ready to rename...")
            os.rename(filename, new_filename)


def create_csv():
    with open('config.json', 'r') as f:
        CONFIG = json.load(f)
    ROOT_PATH = CONFIG["ROOT_PATH"]
    MODEL_DIR = "models-simpleCNN-auc"
    MODEL_DIR = os.path.join(ROOT_PATH, MODEL_DIR)

    print(MODEL_DIR)

    _model_list = os.listdir(MODEL_DIR)
    model_list = []
    for m in _model_list:
        model_list.append(m.replace("model_", "").replace(".h5", ""))
    epochs = [int(_.split('-')[0])+17 for _ in model_list]
    acc = [float(_.split('-')[2]) for _ in model_list]

    print(epochs)
    print(acc)

    with open("./simpleCNN-epoch-auc.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "auc", "score", "date"])
        for i, epoch in enumerate(epochs):
            writer.writerow([epoch, acc[i], "", "20200203"])


def transform_filetree():
    """
    原始train目录下，文件夹名为{label}，下面为文件名为{id}的训练图片
    把它转化为train-*格式文件夹
    """
    with open('config.json', 'r') as f:
        CONFIG = json.load(f)
    ROOT_PATH = CONFIG["ROOT_PATH"]

    TRAIN = os.path.join(ROOT_PATH, "data/train")
    TRAIN_GOOD = os.path.join(TRAIN, "good_0")
    TRAIN_BAD = os.path.join(TRAIN, "bad_1")

    good_sample_files = os.listdir(TRAIN_GOOD)
    size_good_sample = len(good_sample_files)
    mask = list(range(0, size_good_sample))
    np.random.shuffle(mask)
    val_split = int(0.2*size_good_sample)

    os.makedirs(os.path.join(TRAIN, "train/good_0"), exist_ok=True)
    os.makedirs(os.path.join(TRAIN, "val/good_0"), exist_ok=True)
    os.chdir(TRAIN_GOOD)
    for m in tqdm(mask[:val_split]):
        shutil.move(good_sample_files[m], os.path.join(TRAIN, "val/good_0"))
    for m in tqdm(mask[val_split:]):
        shutil.move(good_sample_files[m], os.path.join(TRAIN, "train/good_0"))

    bad_sample_files = os.listdir(TRAIN_BAD)
    size_bad_sample = len(bad_sample_files)
    mask = list(range(0, size_bad_sample))
    np.random.shuffle(mask)
    val_split = int(0.2*size_bad_sample)

    os.makedirs(os.path.join(TRAIN, "train/bad_1"), exist_ok=True)
    os.makedirs(os.path.join(TRAIN, "val/bad_1"), exist_ok=True)
    os.chdir(TRAIN_BAD)
    for m in tqdm(mask[:val_split]):
        shutil.move(bad_sample_files[m], os.path.join(TRAIN, "val/bad_1"))
    for m in tqdm(mask[val_split:]):
        shutil.move(bad_sample_files[m], os.path.join(TRAIN, "train/bad_1"))


def main():
    # create_csv()
    transform_filetree()
    return
    TRAIN = "D:\\DeepLearningData\\semi-conductor-image-classification-first\\data\\train"
    TRAIN_GOOD = os.path.join(TRAIN, "good_0")
    os.chdir(TRAIN_GOOD)
    os.rename('train-good_0-00025ab9befdfbf34891887c1f8d8b4f.jpg',
              '00025ab9befdfbf34891887c1f8d8b4f.jpg')


if __name__ == "__main__":
    main()
