#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Nov-16-20 21:33
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @Link    : http://example.org

import os


def plot_PCA(components_train_bad, components_train_good)():
    import matplotlib.pyplot as plt
    plt.scatter(components_train_bad[:, 0],
                components_train_bad[:, 1], color="r", label="bad", s=0.5)
    plt.scatter(components_train_good[:, 0],
                components_train_good[:, 1], color="g", label="good", s=0.5)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    pass


if __name__ == "__main__":
    main()
