#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Mar-10-20 20:54
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org


import os
import zipfile


DIR = "C:\\Users\\kellyhwong\\GoogleDrive\\DeepLearning\\semi-conductor-image-classification-first-master\\efficientnet_codes"  # 要整个打包的目录
ZIP_FILENAME = os.path.basename(DIR) + ".zip"
ZIP_FILEPATH = os.path.abspath(os.path.join(DIR, os.pardir))
ZIP_FILEPATH = os.path.join(ZIP_FILEPATH, ZIP_FILENAME)


def zip_file():
    # 把整个文件夹内的文件打包
    f = zipfile.ZipFile(ZIP_FILEPATH, 'w', zipfile.ZIP_DEFLATED)
    startdir = DIR
    prefix = os.path.abspath(os.path.join(DIR, os.pardir))
    for dirpath, dirnames, filenames in os.walk(startdir):
        for filename in filenames:
            src_file = os.path.join(dirpath, filename)
            dst_file = src_file.replace(prefix, "")
            print(dst_file)
            f.write(src_file, dst_file)
    f.close()


def unzip_file():
    f = zipfile.ZipFile(ZIP_FILEPATH)
    dst_dir = os.path.abspath(os.path.join(DIR, os.pardir))
    # os.mkdir(os.path.join(dst_dir, "efficientnet_codes"))
    for name in f.namelist():
        f.extract(name, dst_dir)
    f.close()


def main():
    # zip_file()
    unzip_file()


if __name__ == "__main__":
    main()
