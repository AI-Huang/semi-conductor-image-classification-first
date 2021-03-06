# semi-conductor-image-classification-first

semi-conductor-image-classification-first

The in-class image classification contest, in which you need to build a model and recognise bad chips and good chips.

## Data

Semi-conductor images. Two classes: `{"0": "good_0", "1": "bad_1"}`.

Bad chips for example:
![](./fig/bad.png)

Good chips for example:
![](./fig/good.png)

## Approches

主要用了简单 CNN，ResNet20v2，和 ResNet56v2。v2 代表使用的是 ResNet version 2，20 和 56 是层数。

### Loss function

Loss Function 使用 `categorical_crossentropy`, 随后又使用[tensorflow imbalanced_data](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)中方法进行 class weights 加权。

## Results

Current best score is obtained by ResNet56v2:

| Epoch | AUC        | Public Score | Date     |
| ----- | ---------- | ------------ | -------- |
| 109   | **0.9832** | **0.97156**  | 20200225 |

## Engineering contributions

### Confusion matrix

Modified TensorFlow official confusion matrix metrics codes in [confusion_matrix_v2_1_0.py](./keras_fn/confusion_matrix_v2_1_0.py) so that

1.  2D tensor y_pred and y_true input can be accepted;
2.  class_id can be specified when collecting these metrics such as: false positive, false negative, Recall, AUC.

### Test focal loss from Tensorflow Models

https://github.com/tensorflow/models/blob/master/official/vision/keras_cv/losses/focal_loss.py

# License

MIT License
