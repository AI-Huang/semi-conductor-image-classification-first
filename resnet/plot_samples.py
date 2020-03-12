#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-26-20 14:26
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os


def main():
    """Prepare Data Frame"""
    filenames = os.listdir(TRAIN_DATA_DIR)
    categories = []
    for filename in filenames:
        category = filename.split('.')[0]
        if category == 'bad':  # bad 1
            categories.append(1)
        else:  # good 0
            categories.append(0)
    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })

    # print(df.head())
    # print(df.tail())
    # df['category'].value_counts().plot.bar()
    # plt.show()

    """Sample Image"""
    # sample = random.choice(filenames)
    # image = load_img("./data/train/"+sample)
    # plt.imshow(image)
    # plt.show()

    """Prepare data"""
    df["category"] = df["category"].replace({0: 'good', 1: 'bad'})

    """ 这里用来自动划分 train 集和 val 集 """
    train_df, validate_df = train_test_split(
        df, test_size=0.20, random_state=42)
    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)

    # train_df['category'].value_counts().plot.bar()

    total_train = train_df.shape[0]
    total_validate = validate_df.shape[0]

    """ Example Generation """
    example_df = train_df.sample(n=1).reset_index(drop=True)
    example_generator = train_datagen.flow_from_dataframe(
        example_df,
        TRAIN_DATA_DIR,
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical'
    )

    """ Example Generation Ploting """
    # plt.figure(figsize=(12, 12))
    # for i in range(0, 15):
    #     plt.subplot(5, 3, i+1)
    #     for X_batch, Y_batch in example_generator:
    #         image = X_batch[0]
    #         plt.imshow(image)
    #         break
    # plt.tight_layout()
    # plt.show()

    """ Heatmap """


if __name__ == "__main__":
    main()
