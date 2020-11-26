#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-03-20 23:44
# @Update  : Nov-24-20 20:52
# @Author  : Kelly Hwong (you@example.org)
# @Link    : http://example.org

from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, CSVLogger


def learning_reducer(parameter_list):
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    lr_reducer = ReduceLROnPlateau(monitor="auc",
                                   patience=2,
                                   verbose=1,
                                   factor=0.5,
                                   min_lr=0.00001)
    callbacks = [earlystop, lr_reducer]


if __name__ == "__main__":
    main()
