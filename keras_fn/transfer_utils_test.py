#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Nov-17-20 17:05
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @Link    : http://example.org

import os
from resnet import model_depth, resnet_v2
from transfer_utils import ResNetParams, transfer_weights


input_shape = [224, 224, 3]
num_classes = 2


def test_transfer_weights():
    # Prepare Model
    src_param = ResNetParams(n=2, version=2)
    dest_param = ResNetParams(n=6, version=2)

    # Create src model: ResNet20v2
    print("Src model type: %s" % src_param.model_type)
    src_model = resnet_v2(input_shape=input_shape,
                          depth=src_param.depth, num_classes=num_classes)

    # Restore src model's weights
    # MODEL_CKPT_FILE = "ResNet20v2.020-auc-0.9736.h5"
    # filepath = os.path.join(src_param.saves_dir, MODEL_CKPT_FILE)
    # print("loading weights from: %s..." % filepath)
    # src_model.load_weights(filepath)
    # src_weights_list = src_model.get_weights()

    # Create dest model: ResNet56v2
    print("Dest model type: %s" % dest_param.model_type)
    dest_model = resnet_v2(input_shape=input_shape,
                           depth=dest_param.depth, num_classes=num_classes)

    transfer_weights(src_model, dest_model)

    print("Saving new model...")
    filename = f"{dest_param.model_type}.h5"
    dest_model.save_weights(filename)


def main():
    test_transfer_weights()


if __name__ == "__main__":
    main()
