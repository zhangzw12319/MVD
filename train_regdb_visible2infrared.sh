#!/bin/sh
python train.py \
--dataset RegDB \
--data-path "/home/yuki/dataset/RegDB/" \
--optim adam \
--lr 0.00035 \
--device-target GPU \
--gpu 0 \
--pretrain "/home/yuki/code/Mindspore/pretrain/resnet50_ascend_v111_imagenet2012_official_cv_bs32_acc76/resnet50.ckpt" \
--tag "regdb_visible2infrared" \
--loss-func "id+tri" \
--trial "1" \
--regdb_mode "v2i"
