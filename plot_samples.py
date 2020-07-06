#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-26-20 14:26
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import json
import numpy as np
import pandas as pd


def main():
    print("Load Config ...")
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

    print("Prepare data frame...")
    num_train = 30000
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
    df = pd.DataFrame(data=table, columns=columns)
    print(df.head())
    # df['label'].value_counts().plot.bar()
    # plt.show()

    # print("Sample image...")
    # sample = random.choice(filenames)
    # image = load_img("./data/train/"+sample)
    # plt.imshow(image)
    # plt.show()
    # train_df['label'].value_counts().plot.bar()

    # """ Example Generation """
    # example_df = train_df.sample(n=1).reset_index(drop=True)
    # example_generator = train_datagen.flow_from_dataframe(
    #     example_df,
    #     TRAIN_DATA_DIR,
    #     x_col='filename',
    #     y_col='label',
    #     target_size=IMAGE_SIZE,
    #     class_mode='categorical'
    # )

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

    """ Heatmap """


if __name__ == "__main__":
    main()
