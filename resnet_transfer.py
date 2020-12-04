#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Nov-29-20 16:18
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)

import os
import argparse
from datetime import datetime
import tensorflow as tf
from keras_fn.resnet import model_depth, resnet_v2
from keras_fn.transfer_utils import transfer_weights
from utils.gpu_utils import get_gpu_memory, get_available_gpu_indices


def cmd_parser():
    """parse arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--date_time', type=str, dest='date_time',
                        action='store', default=None, help='tmp, manually set date time, for model data save path configuration.')

    # Input parameters
    parser.add_argument('--side_length', type=int, dest='side_length',
                        action='store', default=224, help='side_length, the length of image width and height to be cast to.')

    # Device
    parser.add_argument('--gpu', type=int, dest='gpu',
                        action='store', default=0, help='gpu, the number of the gpu used for experiment.')

    # parser.add_argument('--src_gpu', type=int, dest='src_gpu',
    #                     action='store', default=0, help='src_gpu.')
    # parser.add_argument('--dest_gpu', type=int, dest='dest_gpu',
    #                     action='store', default=0, help='dest_gpu.')

    args = parser.parse_args()

    return args


def main():
    args = cmd_parser()

    # Config GPUs
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[args.gpu:], 'GPU')
    gpu_list = tf.config.list_physical_devices('GPU')

    gpus_memory = get_gpu_memory()
    available_gpu_indices = get_available_gpu_indices(gpus_memory)

    # Data path
    competition_name = "semi-conductor-image-classification-first"
    data_dir = os.path.expanduser(
        f"~/.kaggle/competitions/{competition_name}")

    date_time = args.date_time

    prefix = os.path.join(
        "~", "Documents", "DeepLearningData", competition_name)

    model_type = os.path.join(
        "ResNet20v2_56v2", date_time, "ResNet20v2" + "_pretrain")
    loss_type = "normal"
    subfix = os.path.join(model_type, loss_type)

    ckpt_dir = os.path.expanduser(os.path.join(prefix, "ckpts", subfix))

    # Input paramaters
    SIDE_LENGTH = args.side_length  # default 224
    IMAGE_WIDTH = IMAGE_HEIGHT = SIDE_LENGTH
    image_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
    IMAGE_CHANNELS = 1
    input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
    num_classes = 2

    # Create src model: ResNet20v2
    with tf.device("/device:GPU:" + str(available_gpu_indices[0])):
        src_model = resnet_v2(input_shape=input_shape,
                              depth=model_depth(n=2, version=2), num_classes=num_classes)

    # Load model's last checkpoint if there is any
    latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    assert latest_ckpt is not None
    src_model.load_weights(latest_ckpt)

    # Create dest model: ResNet56v2
    with tf.device("/device:GPU:" + str(available_gpu_indices[1])):
        dest_model = resnet_v2(input_shape=input_shape,
                               depth=model_depth(n=6, version=2), num_classes=num_classes)
    # Do weights transferring
    transfer_weights(src_model, dest_model)

    model_type = os.path.join(
        "ResNet20v2_56v2", date_time, "ResNet56v2" + "_continue")
    loss_type = "normal"
    subfix = os.path.join(model_type, loss_type)

    ckpt_dir = os.path.expanduser(os.path.join(prefix, "ckpts", subfix))
    ckpt_name = "ResNet56v2-transfer-from-ResNet20v2"
    dest_model.save_weights(os.path.join(ckpt_dir, ckpt_name))


if __name__ == "__main__":
    main()
