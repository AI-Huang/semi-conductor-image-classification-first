#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-07-20 11:45
# @Author  : Your Name (you@example.org)
# @Link    : https://www.kaggle.com/c/semi-conductor-image-classification-first

import os
import numpy as np
from resnet import model_depth


def main():
    n = 2
    version = 2
    depth = model_depth(n, version)
    model_type = 'ResNet%dv%d' % (depth, version)

    predict = np.load(model_type + "-predict.npy")
    print(type(predict))
    print(predict[:10, 0])


if __name__ == "__main__":
    main()
