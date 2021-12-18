#!/bin/sh

myfile="run_standalone_train_sysu_indoor_gpu.sh"

if [ ! -f "$myfile" ]; then
    echo "Please first enter MVD/scripts/run_standalone_train and run. Exit..."
    exit 0
fi

cd ..

# Note: --pretrain, --data-path arguments support global path or relative path(starting
#       from project root directory, i.e. /.../DDAG_mindspore/)

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