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
from keras_fn.transfer_utils import transfer_weights
from utils.dir_utils import makedir_exist_ok
from utils.data_utils import data_generators
from utils.gpu_utils import get_gpu_memory, get_available_gpu_indices


def cmd_parser():
    """parse arguments
    """
    parser = argparse.ArgumentParser()

    def string2bool(string):
        """string2bool
        """
        if string not in ["False", "True"]:
            raise argparse.ArgumentTypeError(
                f"""input(={string}) NOT in ["False", "True"]!""")
        if string == "False":
            return False
        elif string == "True":
            return True

    # Input parameters
    parser.add_argument('--side_length', type=int, dest='side_length',
                        action='store', default=224, help='side_length, the length of image width and height to be cast to.')

    # Model parameters
    parser.add_argument('--version', type=int, dest='version',
                        action='store', default=2, help='version, version of ResNet, 1 or 2.')
    parser.add_argument('--n', type=int, dest='n',
                        action='store', default=6, help='n, order of ResNet, 2 or 6.')

    # Experiment type
    parser.add_argument('--exper_type', type=str, dest='exper_type',
                        action='store', default=None, help='exper_type, --exper_type=transfer, to set the training to transfer mode.')

    # Training parameters
    parser.add_argument('--batch_size', type=int, dest='batch_size',
                        action='store', default=32, help='batch_size, e.g. 32.')  # if using ResNet20v2, 32 for Mac, 32, 64, 128 for server
    parser.add_argument('--epochs', type=int, dest='epochs',
                        action='store', default=150, help='training epochs, e.g. 150.')  # training for 150 epochs is sufficient to fit enough
    parser.add_argument('--if_fast_run', type=string2bool, dest='if_fast_run',
                        action='store', default=False, help='if_fast_run, if True, will only train the model for 3 epochs.')

    # Loss
    parser.add_argument('--loss', type=str, dest='loss',
                        action='store', default="bce", help="""loss name, one of ["bce", "focal"].""")
    parser.add_argument('--start_epoch', type=int, dest='start_epoch',
                        action='store', default=0, help='start_epoch, i.e., epoches that have been trained, e.g. 80.')  # 已经完成的训练数
    parser.add_argument('--ckpt', type=str, dest='ckpt',
                        action='store', default="", help='ckpt, model ckpt file.')

    # Focal loss parameters, only necessary when focal loss is chosen
    parser.add_argument('--alpha', type=float, dest='alpha',
                        action='store', default=0.25, help='alpha pamameter for focal loss if it is used.')
    parser.add_argument('--gamma', type=float, dest='gamma',
                        action='store', default=2, help='gamma pamameter for focal loss if it is used.')

    # Device
    parser.add_argument('--visible_gpu_from', type=int, dest='visible_gpu_from',
                        action='store', default=0, help='visible_gpu_from, the first visible gpu index set by tf.config.')
    parser.add_argument('--model_gpu', type=int, dest='model_gpu',
                        action='store', default=None, help='model_gpu, the number of the model_gpu used for experiment.')
    parser.add_argument('--train_gpu', type=int, dest='train_gpu',
                        action='store', default=None, help='train_gpu, the number of the train_gpu used for experiment.')

    # Other parameters
    parser.add_argument('--tmp', type=string2bool, dest='tmp',
                        action='store', default=False, help='tmp, if true, the yielding data during the training process will be saved into a temporary directory.')
    parser.add_argument('--model_type', type=str, dest='model_type',
                        action='store', default=None, help='tmp, manually set model type, for model data save path configuration.')
    parser.add_argument('--date_time', type=str, dest='date_time',
                        action='store', default=None, help='date_time, manually set date time, for model data save path configuration.')
    parser.add_argument('--date_time_first', type=string2bool, dest='date_time_first',
                        action='store', default=False, help="date_time_first, if True, make date_time parameter at first in the directories' suffix.")

    args = parser.parse_args()

    return args


def main():
    args = cmd_parser()

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(
        physical_devices[args.visible_gpu_from:], 'GPU')
    gpus_memory = get_gpu_memory()
    available_gpu_indices = get_available_gpu_indices(gpus_memory)

    # Data path
    competition_name = "semi-conductor-image-classification-first"
    data_dir = os.path.expanduser(
        f"~/.kaggle/competitions/{competition_name}")

    # Model type
    n = args.n  # order of ResNet
    if n == 2:
        if not args.model_gpu:
            model_gpu = available_gpu_indices[0]
            train_gpu = available_gpu_indices[0]
    elif n == 6:
        assert args.batch_size <= 16
        if not args.model_gpu:
            model_gpu = available_gpu_indices[0]
        if not args.train_gpu:
            train_gpu = available_gpu_indices[1]

    version = args.version
    depth = model_depth(n, version)
    model_type = "ResNet%dv%d" % (depth, version)
    if args.exper_type == "transfer":
        if n == 2:
            model_type = model_type + "_pretrain"
        elif n == 6:
            model_type = model_type + "_continue"

    # Experiment time
    if args.date_time == None:
        date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        date_time = args.date_time

    if args.tmp:
        import tempfile
        prefix = tempfile.mkdtemp()
    else:
        prefix = os.path.join(
            "~", "Documents", "DeepLearningData", competition_name)

    loss = args.loss
    if loss == "focal":
        loss_type = "focal"
        subfix = os.path.join(model_type,
                              loss_type, '-'.join(["alpha", f"{args.alpha}"]))
    else:
        loss_type = "normal"
        subfix = os.path.join(model_type, loss_type)

    if args.date_time_first:
        subfix = os.path.join(date_time, subfix)
    else:
        subfix = os.path.join(subfix, date_time)

    if args.exper_type == "transfer":
        subfix = os.path.join("ResNet20v2_56v2", subfix)

    ckpt_dir = os.path.expanduser(os.path.join(prefix, "ckpts", subfix))
    log_dir = os.path.expanduser(os.path.join(prefix, "logs", subfix))
    makedir_exist_ok(ckpt_dir)
    makedir_exist_ok(log_dir)

    # Input paramaters
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
    with tf.device("/device:GPU:" + str(model_gpu)):
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

    with tf.device("/device:GPU:" + str(model_gpu)):
        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=metrics)

    # Load model's last checkpoint if there is any
    latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    if latest_ckpt is not None:
        print("Latest checkpoint loaded!")
        model.load_weights(latest_ckpt)

    # Define callbacks
    from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler, TensorBoard, ModelCheckpoint

    ckpt_name = "%s-epoch-{epoch:03d}-auc-{auc:.4f}" % model_type
    filepath = os.path.join(ckpt_dir, ckpt_name)
    checkpoint_callback = ModelCheckpoint(
        filepath=filepath, monitor="auc", verbose=1, save_weights_only=True)

    csv_logger = CSVLogger(os.path.join(
        log_dir, "training.log.csv"), append=True)

    lr_scheduler = LearningRateScheduler(
        lr_schedule, verbose=1)

    tensorboard_callback = TensorBoard(
        log_dir, histogram_freq=1, update_freq="batch")

    callbacks = [csv_logger, lr_scheduler,
                 checkpoint_callback, tensorboard_callback]

    # Fit model
    epochs = 3 if args.if_fast_run else args.epochs
    with tf.device("/device:GPU:" + str(train_gpu)):
        history = model.fit(
            x=train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            initial_epoch=args.start_epoch
        )


if __name__ == "__main__":
    main()
