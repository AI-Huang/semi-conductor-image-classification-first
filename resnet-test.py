#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-03-20 23:44
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import random
import numpy as np
import pandas as pd
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split
from resnet import model_depth, resnet_v2, lr_schedule
from model import auc

# Training parameters
IF_DATA_AUGMENTATION = True
NUM_CLASSES = 2
IMAGE_WIDTH = IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 1
INPUT_SHAPE = [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS]
# data path
TRAIN_DATA_DIR = "./data/train/"
TEST_DATA_DIR = "./data/test/all_tests"
SAVES_DIR = "./models-ResNet56v2/"

# constants
IF_FAST_RUN = True
EPOCHS_OVER_NIGHT = 50
BATCH_SIZE = 15
# BATCH_SIZE = 32  # orig paper trained all networks with batch_size=128


def main():
    """ Prepare Model """
    n = 6
    version = 2
    depth = model_depth(n, version)
    model_type = 'ResNet%dv%d' % (depth, version)
    model = resnet_v2(input_shape=INPUT_SHAPE, depth=depth, num_classes=2)
    model.compile(loss='categorical_crossentropy',
                  #   optimizer=Adam(learning_rate=lr_schedule(0)),
                  optimizer='adam',
                  metrics=['accuracy', auc])
    model.summary()
    print(model_type)
    print("loading weights")
    model.load_weights(os.path.join(SAVES_DIR, "ResNet56v2.001-auc-0.9092.h5"))

    """ Prepare Data Frame """
    filenames = os.listdir(TRAIN_DATA_DIR)
    random.shuffle(filenames)
    categories = []
    for filename in filenames:
        category = filename.split('.')[0]
        if category == 'good':
            categories.append(1)
        else:
            categories.append(0)
    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })
    df["category"] = df["category"].replace({0: 'good', 1: 'bad'})

    """ 这里用来自动划分 train 集和 val 集 """
    train_df, validate_df = train_test_split(
        df, test_size=0.20, random_state=42)
    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)

    """Traning Generator"""
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        TRAIN_DATA_DIR,
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        class_mode='categorical',
        batch_size=BATCH_SIZE
    )

    """Prepare Testing Data"""
    test_filenames = os.listdir(TEST_DATA_DIR)
    test_df = pd.DataFrame({
        'filename': test_filenames
    })
    # TODO 不知道为什么叫 nb_samples
    nb_samples = test_df.shape[0]

    test_gen = ImageDataGenerator(rescale=1./255)

    """Create Testing Generator"""
    test_generator = test_gen.flow_from_dataframe(
        test_df,
        TEST_DATA_DIR,
        x_col='filename',
        y_col=None,
        class_mode=None,
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        shuffle=False
    )  # Found 12500 images.

    """ Predict """
    # model = load_model("./models/model_24-auc-0.9639.h5")

    # predict 要时间
    import time
    start = time.clock()

    predict = model.predict_generator(
        test_generator, steps=np.ceil(nb_samples / BATCH_SIZE))

    elapsed = (time.clock() - start)
    print("Prediction time used:", elapsed)

    np.save(model_type + "-predict.npy", predict)

    """
    # numpy average max
    # test_df['category'] = np.argmax(predict, axis=-1)

    # We will convert the predict category back into our generator classes by using train_generator.class_indices. It is the classes that image generator map while converting data into computer vision
    label_map = dict((v, k) for k, v in train_generator.class_indices.items())
    test_df['category'] = test_df['category'].replace(label_map)

    # From our prepare data part. We map data with {1: 'dog', 0: 'cat'}. Now we will map the result back to dog is 1 and cat is 0
    test_df['category'] = test_df['category'].replace({'dog': 1, 'cat': 0})
    test_df['category'].value_counts().plot.bar()

    plt.show()
    """

    # predict 的第1列是 good
    test_df['category'] = predict[:, 0]
    print("Predict Samples: ")
    print(type(test_df))
    print(test_df.head(10))

    """See predicted result with sample images"""
    """
    sample_test = test_df.head(18)
    sample_test.head()
    plt.figure(figsize=(12, 24))
    for index, row in sample_test.iterrows():
        filename = row['filename']
        category = row['category']
        img = load_img("../input/test1/test1/"+filename, target_size=IMAGE_SIZE)
        plt.subplot(6, 3, index+1)
        plt.imshow(img)
        plt.xlabel(filename + '(' + "{}".format(category) + ')')
    plt.tight_layout()
    plt.show()
    """

    """Submission"""
    submission_df = test_df.copy()
    submission_df['id'] = submission_df['filename'].str.split('.').str[0]
    submission_df['label'] = submission_df['category']
    submission_df.drop(['filename', 'category'], axis=1, inplace=True)
    submission_df.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    main()
