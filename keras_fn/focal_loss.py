#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jul-05-20 12:21
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @RefLink : https://github.com/fudannlp16/focal-loss/blob/master/focal_loss.py

# requirements
# tensorflow==2.2.0
# keras: 2.3.0-tf

import os
import tensorflow as tf


def binary_focal_loss(gamma=2, alpha=0.25):
    """
    Binary form of focal loss.
    适用于二分类问题的focal loss

    focal_loss(p_t) = -alpha_t * (1-p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1-p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)

        p_t = y_true*y_pred + (K.ones_like(y_true)-y_true) * \
            (K.ones_like(y_true)-y_pred) + K.epsilon()
        focal_loss = - alpha_t * \
            K.pow((K.ones_like(y_true)-p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)
    return binary_focal_loss_fixed


def focal_loss_softmax(labels, logits, gamma=2):
    """
    Computer focal loss for multi classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size,num_classes].
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred = tf.nn.softmax(logits, dim=-1)  # [batch_size,num_classes]
    labels = tf.one_hot(labels, depth=y_pred.shape[1])
    L = -labels*((1-y_pred)**gamma)*tf.log(y_pred)
    L = tf.reduce_sum(L, axis=1)
    return L


def main():
    pass


if __name__ == "__main__":
    main()
