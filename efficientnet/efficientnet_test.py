#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Mar-11-20 21:07
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf

import model_builder_factory
import preprocessing
import utils

import os
import json
import pandas as pd
import numpy as np

flags.DEFINE_string('model_name', 'efficientnet-b0', 'Model name to eval.')
flags.DEFINE_string('runmode', 'examples',
                    'Running mode: examples or imagenet')
flags.DEFINE_string(
    'imagenet_eval_glob', None, 'Imagenet eval image glob, '
    'such as /imagenet/ILSVRC2012*.JPEG')
flags.DEFINE_string(
    'imagenet_eval_label', None, 'Imagenet eval label file path, '
    'such as /imagenet/ILSVRC2012_validation_ground_truth.txt')
flags.DEFINE_string('ckpt_dir', './archive-test',
                    'Checkpoint folders')
flags.DEFINE_boolean('enable_ema', True, 'Enable exponential moving average.')
flags.DEFINE_string('export_ckpt', None, 'Exported ckpt for eval graph.')
flags.DEFINE_string('example_img', './test.img',  # "D:\\DeepLearningData\\semi-conductor-image-classification-first\\semi-conductor-image-classification-first\\test\\all_tests"
                    'Filepath for a single example image.')
flags.DEFINE_string('tests_dir', "D:\\DeepLearningData\\semi-conductor-image-classification-first\\data\\origin\\test\\all_tests",
                    'tests_dir.')
flags.DEFINE_string('labels_map_file', './labels_map.txt',
                    'Labels map from label id to its meaning.')
flags.DEFINE_bool('include_background_label', False,
                  'Whether to include background as label #0')
flags.DEFINE_bool('advprop_preprocessing', False,
                  'Whether to use AdvProp preprocessing.')
flags.DEFINE_integer('num_images', -1,
                     'Number of images to eval. Use -1 to eval all images.')


def f123():
    """ Load Config """
    with open('./config/config.json', 'r') as f:
        CONFIG = json.load(f)
    ROOT_PATH = CONFIG["ROOT_PATH"]
    TEST_DATA_DIR = CONFIG["TEST_DATA_DIR"]
    TEST_DATA_DIR = os.path.join(ROOT_PATH, TEST_DATA_DIR)

    """ 测试模型 """

    # 可以测试单张图片

    """ Prepare Testing Data """
    test_filenames = os.listdir(TEST_DATA_DIR)
    test_df = pd.DataFrame({
        'filename': test_filenames
    })
    nb_samples = test_df.shape[0]

    """ 提交submission """
    # predict 第 1 列，是不是 bad
    test_df['label'] = predict[:, 0]
    print("Predict Samples: ")
    print(type(test_df))
    print(test_df.head(10))

    submission_df = test_df.copy()
    submission_df['id'] = submission_df['filename'].str.split('.').str[0]
    submission_df['label'] = submission_df['label']
    submission_df.drop(['filename', 'label'], axis=1, inplace=True)
    submission_df.to_csv('./submissions/submission-%s.csv' %
                         MODEL_CKPT, index=False)


class EvalCkptDriver(utils.EvalCkptDriver):
    """A driver for running eval inference."""

    def build_model(self, features, is_training):
        """Build model with input features."""
        tf.logging.info(self.model_name)
        model_builder = model_builder_factory.get_model_builder(
            self.model_name)

        if self.advprop_preprocessing:
            # AdvProp uses Inception preprocessing.
            features = features * 2.0 / 255 - 1.0
        else:
            features -= tf.constant(
                model_builder.MEAN_RGB, shape=[1, 1, 3], dtype=features.dtype)
            features /= tf.constant(
                model_builder.STDDEV_RGB, shape=[1, 1, 3], dtype=features.dtype)
        logits, _ = model_builder.build_model(
            features, self.model_name, is_training)
        probs = tf.nn.softmax(logits)
        probs = tf.squeeze(probs)
        return probs

    def get_preprocess_fn(self):
        """Build input dataset."""
        return preprocessing.preprocess_image


def get_eval_driver(model_name,
                    include_background_label=False,
                    advprop_preprocessing=False):
    """Get a eval driver."""
    image_size = model_builder_factory.get_model_input_size(model_name)
    return EvalCkptDriver(
        model_name=model_name,
        batch_size=1,
        image_size=image_size,
        include_background_label=include_background_label,
        advprop_preprocessing=advprop_preprocessing)


# FLAGS should not be used before main.
FLAGS = flags.FLAGS


def main(unused_argv):
    logging.set_verbosity(logging.ERROR)
    driver = get_eval_driver(FLAGS.model_name, FLAGS.include_background_label,
                             FLAGS.advprop_preprocessing)
    if FLAGS.runmode == 'examples':
        # Run inference for an example image.
        # [FLAGS.example_img]
        test_imgs = os.listdir(FLAGS.tests_dir)
        test_imgs = [os.path.join(FLAGS.tests_dir, _) for _ in test_imgs]
        pred_idx, pred_prob = driver.eval_example_images(FLAGS.ckpt_dir,
                                                         test_imgs,
                                                         FLAGS.labels_map_file, FLAGS.enable_ema,
                                                         FLAGS.export_ckpt)
        pred_prob = np.asarray(pred_prob)
        print("pred_prob", pred_prob)

        test_df = pd.DataFrame({
            'filename': os.listdir(FLAGS.tests_dir)
        })
        # pred_prob 第 2 列，是不是 bad_1
        test_df['label'] = pred_prob[:, 1]

        print(test_df.head(10))

        """ 提交submission """
        submission_df = test_df.copy()
        submission_df['id'] = submission_df['filename'].str.split('.').str[0]
        submission_df['label'] = submission_df['label']
        submission_df.drop(['filename', 'label'], axis=1, inplace=True)
        submission_df.to_csv(
            './submissions/submission-EfficientNet.csv', index=False)

    elif FLAGS.runmode == 'imagenet':
        # Run inference for imagenet.
        driver.eval_imagenet(FLAGS.ckpt_dir, FLAGS.imagenet_eval_glob,
                             FLAGS.imagenet_eval_label, FLAGS.num_images,
                             FLAGS.enable_ema, FLAGS.export_ckpt)
    else:
        print('must specify runmode: examples or imagenet')


if __name__ == '__main__':
    app.run(main)
