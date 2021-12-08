#!/bin/sh

myfile="train_sysu_all.sh"

if [ ! -f "$myfile" ]; then
    echo "Please first enter MVD/scripts/run_standalone_train and run. Exit..."
    exit 0
fi

cd ../..

python train.py \
--dataset "SYSU" \
--data-path "/home/shz/pytorch/data/sysu" \
--optim adam \
--lr 0.00035 \
--device-target GPU \
--gpu 1 \
--pretrain "/home/shz/pytorch/zzw/DDAG_mindspore/model/pretrain/resnet50_ascend_v111_imagenet2012_official_cv_bs32_acc76/resnet50.ckpt" \
--tag "sysu_all" \
--loss-func "id+tri" \
--sysu_mode "all" \
--epoch 80