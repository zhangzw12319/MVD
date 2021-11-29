#!/bin/sh
python train.py \
--dataset SYSU \
--data-path "/home/yuki/dataset/SYSU/" \
--optim adam \
--lr 0.00035 \
--device-target GPU \
--gpu 0 \
--pretrain "/home/yuki/code/Mindspore/pretrain/resnet50_ascend_v111_imagenet2012_official_cv_bs32_acc76/resnet50.ckpt" \
--tag "sysu_indoor" \
--loss-func "id+tri" \
--sysu_mode "indoor"