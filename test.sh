#!/bin/sh
python test.py \
--dataset SYSU \
--data-path "/home/yuki/dataset/SYSU/" \
--device-target GPU \
--gpu 0 \
--resume "logs/toy/training/epoch_18_rank1_2.76_mAP_4.15_SYSU_batch-size_2*8*4=64_adam_lr_0.00035_loss-func_id+tri_master.ckpt" \
--tag "toy"