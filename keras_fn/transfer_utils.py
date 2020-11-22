#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-01-20 06:51
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)
# @Link    : https://stackoverflow.com/questions/43702323/how-to-load-only-specific-weights-on-keras


from resnet import model_depth


class ResNetParams(object):
    def __init__(self, n=2, version=2):
        self.n = n
        self.version = version
        self.depth = model_depth(self.n, self.version)
        self.model_type = "ResNet%dv%d" % (self.depth, self.version)


def print_layers(model):
    """print_layers, print model weights' shape
    Input:
        model: Keras model.
    """
    for i in range(len(model.layers)):
        print("Printing layer shape: %d" % i, model.layers[i])
        weights = model.layers[i].get_weights()
        for weight in weights:  # Layer type
            print(weight.shape)


def transfer_weights(src_model, dest_model):
    """transfer_weights, transfer weights for two types of ResNet.
    Inputs:
        src_model: a ResNet20v2 Keras model.
        dest_model: a ResNet56v2 Keras model.
    Return:
        None
    """
    # ingore the first layer Input()
    # layer 1-24 to 1-24
    for i in range(1, 24):
        dest_model.layers[i].set_weights(src_model.layers[i].get_weights())
    print("Partially load weights from layer 1-24 successfully!")

    # layer 25-45 to 65-85
    for i in range(25, 45):
        dest_model.layers[i+40].set_weights(src_model.layers[i].get_weights())
    print("Partially load weights from layer 25-45 successfully!")

    # layer 46-65 to 126-145
    for i in range(46, 65):
        dest_model.layers[i+80].set_weights(src_model.layers[i].get_weights())
    print("Partially load weights from layer 46-65 successfully!")

    # 69 to 189
    dest_model.layers[69+120].set_weights(src_model.layers[69].get_weights())
    print("Partially load weights from layer 69 successfully!")


def main():
    pass


if __name__ == "__main__":
    main()
