#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jul-03-20 10:02
# @Author  : Kelly Hwong (you@example.org)
# @Link    : http://example.org

import os
import re
import numpy as np
import pandas as pd


def match_filename(parameter_list):
    pass


def test2():
    # filename = "ResNet56v2.start-8-epoch-017-auc_good_0-0.9887-auc_bad_1-0.9842.h5"
    # "ResNet56v2.000-auc-0.9092.h5" -> "ResNet56v2-epoch-000-auc-0.9092.h5"
    filename = "ResNet56v2.000-auc-0.9092.h5"
    # m = re.search(r"\.", filename)
    # m = re.match(r"\S+\.", filename)  # Greedy match
    m = re.match(r"\S+?\.", filename)  # None greedy match
    print(f"filename: {filename}")
    print(m)
    idx = m.span()[1]
    print(idx)  # right index of the last char
    print(filename[idx-1])  # should be idx-1


def test3():
    # "ResNet56v2.000-auc-0.9092.h5" -> "ResNet56v2-epoch-000-auc-0.9092.h5"
    # "." -> "-epoch-"
    filename = "ResNet56v2.102-auc-0.9807.h5"
    # m = re.search(r"\.", filename)
    # m = re.match(r"\S+\.", filename)  # Greedy match
    m = re.match(r"\S+?\.", filename)  # None greedy match
    print(f"filename: {filename}")
    print(m)
    idx = m.span()[1]
    print(idx)  # right index of the last char
    print(filename[idx-1])  # should be idx-1
    filename = filename[:idx-1] + "-epoch-" + filename[idx-1:]
    print(f"filename: {filename}")


def re_rename1(filename):
    """
    "ResNet56v2.000-auc-0.9092.h5" -> "ResNet56v2-epoch-000-auc-0.9092.h5"
    在这种样子的文件名之间插入"-epoch-"
    """
    # check filename: ResNet56v2 一个dot 加几个数字
    m = re.match(r"ResNet56v2\.[0-9]{3}", filename)
    print(m)
    if m:
        m = re.match(r"ResNet56v2\.", filename)
        print(m)
        idx = m.span()[1]
        # print(idx)  # right index of the last char
        # print(filename[idx-1])  # should be idx-1
        filename_new = filename[:idx-1] + "-epoch-" + filename[idx:]
        print(f"file {filename} should renamed to {filename_new}.")
        return filename_new
    else:
        print(f"file {filename} not renamed.")


def re_rename2(filename):
    """
    "ResNet56v2.start-0-epoch-001-auc_good_0-0.9837-auc_bad_1-0.9829.h5" -> 
    提取出 start-0-epoch-001 中的 001，并与 109 相加 ->
    "ResNet56v2-epoch-110-auc_good_0-0.9837-auc_bad_1-0.9829.h5"
    """
    # 检查文件名，能匹配到 "start" 即可
    m = re.match(r"ResNet56v2.start", filename)
    if m:
        print(f"file {filename} needs to be renamed.")
        # 匹配 "-epoch-" 后的三位数字
        idx = m.span()[1]
        filename_new = filename[:idx-1] + "-epoch-" + filename[idx:]
        print(f"filename_new: {filename_new}")
    else:
        print(f"file {filename} no need to be renamed.")
        pass


def test_rename2():
    filename_tests = ["ResNet56v2-epoch-093-auc-0.9725.h5",
                      "ResNet56v2.start-0-epoch-001-auc_good_0-0.9837-auc_bad_1-0.9829.h5"]
    for f in filename_tests:
        f_new = re_rename2(filename=f)


def test_rename1(filename_tests, src_dst):
    # filename_tests = ["ResNet56v2.000-auc-0.9092.h5",
    #                   "ResNet56v2.start-0-epoch-001-auc_good_0-0.9837-auc_bad_1-0.9829.h5"]
    for f in filename_tests:
        f_new = re_rename1(filename=f)
        if f_new:
            os.rename(os.path.join(src_dst, f),
                      os.path.join(src_dst, f_new))


def test_rename3(filename_tests, src_dst):
    """放弃正则
    ResNet56v2.start-8-epoch-016-auc_good_0-0.9886-auc_bad_1-0.9839
    """
    for f in filename_tests:
        if f.find("start") >= 0:
            r = filter(str.isdigit, f)
            l = [_ for _ in r]
            start_from = l[3]
            middle_epoch = "".join(l[4:7])  # 从 epoch 109 之后又训练的 epoch 计数
            epoch_num = 109 + int(middle_epoch) + int(start_from)
            epoch_num = str(epoch_num)
            print(epoch_num)
            f_new = "ResNet56v2" + "-epoch-" + epoch_num + f[28:]
            print(f"f_new: {f_new}")
            os.rename(os.path.join(src_dst, f),
                      os.path.join(src_dst, f_new))


def main():
    ROOT_PATH = "D:\\DeepLearningData\\semi-conductor-image-classification-first"
    print(f"ROOT_PATH: {ROOT_PATH}")
    MODEL_TYPE = 'ResNet%dv%d' % (56, 2)
    SAVES_DIR = os.path.join(ROOT_PATH, "models-%s/" % MODEL_TYPE)
    print(f"SAVES_DIR: {SAVES_DIR}")
    filename_tests = os.listdir(SAVES_DIR)
    # test_rename1(filename_tests, SAVES_DIR)
    # test_rename3(filename_tests, SAVES_DIR)

    results = np.zeros(shape=(150, 4))
    results[:, 0] = np.linspace(0, 149, num=150)
    # metrics = ["auc", "val_accuracy", "auc_good_0", "auc_bad_1"]
    metrics = ["val_accuracy", "auc_good_0", "auc_bad_1"]
    for f in filename_tests:
        head = "ResNet56v2-epoch-"
        epoch = f[len(head):len(head)+3]
        epoch = int(epoch)
        print(f"epoch: {epoch}; ", end="")
        results[epoch, 0] = epoch
        if epoch == 0:  # auc
            m = "auc"
            idx = f.find(m)
            if idx >= 0:
                metric = f[idx+1+len(m):idx+1+len(m)+6]
            print(f"metric: {m}={metric}")
            results[epoch, 2] = metric
        elif epoch in range(1, 17+1):  # val_accuracy
            m = metrics[1]
            idx = f.find(m)
            if idx >= 0:
                metric = f[idx+1+len(m):idx+1+len(m)+6]
            print(f"metric: {m}={metric}")
            results[epoch, 1] = metric
        elif epoch in range(18, 109+1):  # auc
            m = metrics[0]
            idx = f.find(m)
            if idx >= 0:
                metric = f[idx+1+len(m):idx+1+len(m)+6]
            print(f"metric: {m}={metric}")
            results[epoch, 1] = metric
        elif epoch in range(110, 149+1):
            m1, m2 = metrics[1], metrics[2]
            idx = f.find(m1)
            if idx >= 0:
                metric = f[idx+1+len(m1):idx+1+len(m1)+6]
                print(f"metric: {m1}={metric},", end="")
                results[epoch, 2] = metric
            idx = f.find(m2)
            if idx >= 0:
                metric = f[idx+1+len(m2):idx+1+len(m2)+6]
                print(f"{m2}={metric}")
                results[epoch, 3] = metric

    columns = ["epoch"] + metrics
    df = pd.DataFrame(data=results, columns=columns)
    print(df)
    df.to_csv("./log/training.log.ResNet56v2_origin.csv", index=False)


if __name__ == "__main__":
    main()
