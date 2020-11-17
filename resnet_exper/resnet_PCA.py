#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Mar-16-20 20:10
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import json
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA


def calc_PCA():
    with open('./config/config_origin.json', 'r') as f:
        CONFIG = json.load(f)
    ROOT_PATH = CONFIG["ROOT_PATH"]
    MODEL_TYPE = 'ResNet%dv%d' % (56, 2)
    FEATURE_DIR = os.path.join(ROOT_PATH, "features")
    FEATURE_DIR = os.path.join(FEATURE_DIR, "models-%s/" % MODEL_TYPE)

    features_train_bad = np.load(os.path.join(
        FEATURE_DIR, "features_train_bad.npy"))
    features_train_good = np.load(os.path.join(
        FEATURE_DIR, "features_train_good.npy"))
    features_train = np.concatenate((features_train_bad, features_train_good))

    ipca = IncrementalPCA(n_components=2, batch_size=1000)
    ipca.fit(features_train)  # fit with ALL data
    # components_train = ipca.transform(features_train)
    components_train_bad = ipca.transform(features_train_bad)
    components_train_good = ipca.transform(features_train_good)
    # print(components_train.shape) # (30000, 2)
    np.save(os.path.join(FEATURE_DIR, "components_train_bad.npy"),
            components_train_bad)
    np.save(os.path.join(FEATURE_DIR, "components_train_good.npy"),
            components_train_good)

    import matplotlib.pyplot as plt
    plt.scatter(components_train_bad[:, 0],
                components_train_bad[:, 1], color="r")
    plt.scatter(components_train_good[:, 0],
                components_train_good[:, 1], color="g")
    plt.show()


def main():
    from utils.visualization import plot_PCA

    with open('./config/config_origin.json', 'r') as f:
        CONFIG = json.load(f)
    ROOT_PATH = CONFIG["ROOT_PATH"]
    MODEL_TYPE = 'ResNet%dv%d' % (56, 2)
    FEATURE_DIR = os.path.join(ROOT_PATH, "features")
    FEATURE_DIR = os.path.join(FEATURE_DIR, "models-%s/" % MODEL_TYPE)

    components_train_bad = np.load(os.path.join(
        FEATURE_DIR, "components_train_bad.npy"))
    components_train_good = np.load(os.path.join(
        FEATURE_DIR, "components_train_good.npy"))
    plot_PCA(components_train_bad, components_train_good)


if __name__ == "__main__":
    main()
