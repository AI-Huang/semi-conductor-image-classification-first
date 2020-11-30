#!/bin/sh
# @Date    : Nov-28-20 18:18
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)

date_time=$(date "+%Y%m%d-%H%M%S")

transfer_from=50
continuous_epochs=150

python ./resnet_train.py --exper_type=transfer --n=2 --batch_size=32 --epochs=${transfer_from} --date_time=${date_time} --date_time_first=True
python ./resnet_transfer.py  --date_time=${date_time}
python ./resnet_train.py --exper_type=transfer --n=6 --batch_size=16 --epochs=${continuous_epochs} --date_time=${date_time} --date_time_first=True
