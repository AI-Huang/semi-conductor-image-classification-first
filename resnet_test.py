#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-03-20 23:44
# @Author  : Kelly Hwong (you@example.org)
# @Link    : http://example.org

import os
import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
from keras_fn.resnet import model_depth, resnet_v2, lr_schedule
import keras_fn.confusion_matrix_v2_1_0 as confusion_matrix
from keras_fn import model_config
from utils.data_utils import get_test_generator


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
    parser.add_argument('--positive_class', type=str, dest='positive_class',
                        action='store', default="bad_1", help='positive_class, the class regarded as positive samples, e.g. good_0 or bad_1.')

    parser.add_argument('--alpha', type=float, dest='alpha',
                        action='store', default=0.99, help='alpha for focal loss if this loss is used.')

    args = parser.parse_args()

    return args


def main():
    args = cmd_parser()

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[args.gpu:], 'GPU')

    classes = ["good_0", "bad_1"]
    if args.positive_class == "good_0":
        classes = ["bad_1", "good_0"]

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

    # Input parameters
    SIDE_LENGTH = args.side_length  # default 224
    IMAGE_WIDTH = IMAGE_HEIGHT = SIDE_LENGTH
    image_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
    IMAGE_CHANNELS = 1
    input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
    num_classes = 2

    # Data loaders
    test_generator, test_df = get_test_generator(
        data_dir, target_size=image_size, batch_size=args.batch_size)

    # Create model
    model = resnet_v2(input_shape=input_shape,
                      depth=depth, num_classes=2)

    # Compile model
    loss = args.loss
    if loss == "bce":
        from tensorflow.keras.losses import BinaryCrossentropy
        loss = BinaryCrossentropy()
    elif loss == "cce":
        from tensorflow.keras.losses import CategoricalCrossentropy
        loss = CategoricalCrossentropy()
    else:
        raise ValueError(
            "Input error for args.loss, please type 'bce' or 'cce'")

    from tensorflow.keras.optimizers import Adam
    optimizer = Adam(learning_rate=lr_schedule(args.start_epoch))
    metrics = model_config.get_confusion_matrix_metrics(
        class_id=classes.index("bad_1"))

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

    # Load model checkpoint to be tested
    model_ckpt_file = model_type
    assert os.path.isfile(model_ckpt_file)
    print("Model ckpt found! Loading...:\n%s" % model_ckpt_file)
    model.load_weights(model_ckpt_file)

    # Start prediction
    import time
    start = time.perf_counter()
    print("Start testing...")
    predict = model.predict(
        test_generator,
        workers=4, verbose=1
    )
    elapsed = (time.perf_counter() - start)
    print("Prediction time used:", elapsed)

    np.save(os.path.join("predicts", model_type+"-predict.npy"), predict)

    # Prepare submission
    # predict 第 1 列，是 bad_1 的概率
    test_df['label'] = predict[:, classes.index("bad_1")]
    print("Predict Samples: ")
    print(type(test_df))
    print(test_df.head(10))
    submission_df = test_df.copy()
    submission_df['id'] = submission_df['filename'].str.split('.').str[0]
    submission_df.drop(['filename'], axis=1, inplace=True)
    submission_df = submission_df[list(("id", "label"))]
    submission_df.to_csv(
        f"./submissions/submission-{model_type}-{date_time}.csv", index=False)


if __name__ == "__main__":
    main()
