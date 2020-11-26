#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Aug-06-20 14:33
# @Update  : Nov-25-20 03:15
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)


import os
import errno
import six
import shutil

# in tensorflow.python.keras.utils import data_utils


def makedir_exist_ok(dirpath):
    """makedir_exist_ok compatible for both Python 2 and Python 3
    """
    if six.PY3:
        os.makedirs(
            dirpath, exist_ok=True)  # pylint: disable=unexpected-keyword-arg
    else:
        # Python 2 doesn't have the exist_ok arg, so we try-except here.
        try:
            os.makedirs(dirpath)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def delete_experiment_data(model_type, exper_id, prefix="."):
    """delete_experiment_data
    Delete experiment data according to its model_type and exper_id.
    Inputs:
        prefix: experiment data root path, where ckpts and logs directories should exist
    """
    ckpt_dir_prefix = os.path.expanduser(os.path.join(prefix, "ckpts"))
    log_dir_prefix = os.path.expanduser(os.path.join(prefix, "logs"))

    ckpt_dir = os.path.join(ckpt_dir_prefix, model_type)
    log_dir = os.path.join(log_dir_prefix, model_type)

    dir_to_remove = os.path.join(ckpt_dir, exper_id)
    # try:
    print(f"removing: {dir_to_remove}")
    shutil.rmtree(dir_to_remove, ignore_errors=True)

    dir_to_remove = os.path.join(log_dir, exper_id)
    # try:
    print(f"removing: {dir_to_remove}")
    shutil.rmtree(dir_to_remove, ignore_errors=True)
