#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-03-20 23:44
# @Update  : Nov-19-20 15:34
# @Author  : Kelly Hwong (you@example.org)
# @Link    : http://example.org
"""Training with ResNet
Typical usage:
    python resnet_train.py
Environments:
    TensorFlow version: 2.x
"""

import os
import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
from keras_fn.resnet import model_depth, resnet_v2, lr_schedule
import keras_fn.confusion_matrix_v2_1_0 as confusion_matrix
from keras_fn import model_config
from utils.dir_utils import makedir_exist_ok
from utils.data_utils import data_generators


def cmd_parser():
    """parse arguments
    """
    parser = argparse.ArgumentParser()

    # Device
    parser.add_argument('--gpu', type=int, dest='gpu',
                        action='store', default=0, help='gpu, the number of the gpu used for experiment.')
    # Input parameters
    parser.add_argument('--side_length', type=int, dest='side_length',
                        action='store', default=224, help='side_length, the length of image width and height to be cast to.')

    # Model parameters
    parser.add_argument('--version', type=int, dest='version',
                        action='store', default=2, help='version, version of ResNet, 1 or 2.')
    parser.add_argument('--n', type=int, dest='n',
                        action='store', default=6, help='n, order of ResNet, 2 or 6.')

    # Training parameters
    parser.add_argument('--batch_size', type=int, dest='batch_size',
                        action='store', default=32, help='batch_size, e.g. 32.')  # 32 for Mac, 64, 128 for server
    parser.add_argument('--epochs', type=int, dest='epochs',
                        action='store', default=150, help='training epochs, e.g. 150.')  # training 150 epochs to fit enough

    # Loss
    parser.add_argument('--loss', type=str, dest='loss',
                        action='store', default="bce", help="loss name, one of  'bce' and 'cce'.")

    parser.add_argument('--start_epoch', type=int, dest='start_epoch',
                        action='store', default=0, help='start_epoch, i.e., epoches that have been trained, e.g. 80.')  # 已经完成的训练数
    parser.add_argument('--ckpt', type=str, dest='ckpt',
                        action='store', default="", help='ckpt, model ckpt file.')

    # Focal loss paramaters
    parser.add_argument('--alpha', type=float, dest='alpha',
                        action='store', default=0.75, help='alpha pamameter for focal loss if it is used.')
    parser.add_argument('--gamma', type=float, dest='gamma',
                        action='store', default=2, help='gamma pamameter for focal loss if it is used.')

    args = parser.parse_args()

    return args


def main():
    args = cmd_parser()

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[args.gpu:], 'GPU')

    if_fast_run = False

    # Data path
    competition_name = "semi-conductor-image-classification-first"
    data_dir = os.path.expanduser(
        f"~/.kaggle/competitions/{competition_name}")

    # experiment time
    n = args.n
    version = args.version
    depth = model_depth(n, version)
    model_type = 'ResNet%dv%d' % (depth, version)

    date_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    prefix = os.path.join(
        "~", "Documents", "DeepLearningData", competition_name)
    subfix = os.path.join(model_type, date_time)
    ckpt_dir = os.path.expanduser(os.path.join(prefix, "ckpts", subfix))
    log_dir = os.path.expanduser(os.path.join(prefix, "logs", subfix))
    makedir_exist_ok(ckpt_dir)
    makedir_exist_ok(log_dir)

    # Input parameters
    SIDE_LENGTH = args.side_length  # default 224
    IMAGE_WIDTH = IMAGE_HEIGHT = SIDE_LENGTH
    image_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
    IMAGE_CHANNELS = 1
    input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
    num_classes = 2

    # Data loaders
    classes = ["good_0", "bad_1"]
    train_generator, validation_generator = data_generators(
        data_dir, target_size=image_size, batch_size=args.batch_size, classes=classes)

    print("Train class_indices: ", train_generator.class_indices)
    print("Val class_indices: ", validation_generator.class_indices)

    # Create model
    model = resnet_v2(input_shape=input_shape,
                      depth=depth, num_classes=2)

    # Compile model
    loss = args.loss
    if loss == "bce":
        from tensorflow.keras.losses import BinaryCrossentropy
        loss = BinaryCrossentropy()
    elif loss == "focal":
        from keras_fn.focal_loss import BinaryFocalLoss
        loss = BinaryFocalLoss(alpha=args.alpha, gamma=args.gamma)
    else:
        raise ValueError(
            """loss parameter must be one of ["bce", "focal"].""")

    from tensorflow.keras.optimizers import Adam
    optimizer = Adam(learning_rate=lr_schedule(args.start_epoch))
    metrics = model_config.get_confusion_matrix_metrics(
        class_id=classes.index("bad_1"))

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

    # Define callbacks
    from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler, TensorBoard, ModelCheckpoint

    ckpt_name = "%s-epoch-{epoch:03d}-auc-{auc:.4f}.h5" % model_type
    filepath = os.path.join(ckpt_dir, ckpt_name)
    checkpoint = ModelCheckpoint(
        filepath=filepath, monitor="auc", verbose=1)

    csv_logger = CSVLogger(os.path.join(
        log_dir, "training.log.csv"), append=True)

    lr_scheduler = LearningRateScheduler(
        lr_schedule, verbose=1)

    tensorboard_callback = TensorBoard(
        log_dir, histogram_freq=1, update_freq="batch")

    callbacks = [csv_logger, lr_scheduler, checkpoint, tensorboard_callback]

    # Fit model
    epochs = 3 if if_fast_run else args.epochs
    history = model.fit(
        x=train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        initial_epoch=args.start_epoch
    )


if __name__ == "__main__":
    main()
