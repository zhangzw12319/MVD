#!/bin/sh

myfile="run_standalone_train_sysu_all_gpu.sh"

if [ ! -f "$myfile" ]; then
    echo "Please first enter MVD/scripts/run_standalone_train and run. Exit..."
    exit 0
fi

cd ..

# Note: --pretrain, --data-path arguments support global path or relative path(starting
#       from project root directory, i.e. /.../DDAG_mindspore/)

python train.py \
--dataset "SYSU" \
--data-path "/home/shz/pytorch/data/sysu" \
--optim adam \
--lr 0.00035 \
--device-target GPU \
--gpu 2 \
--pretrain "/home/shz/pytorch/zzw/DDAG_mindspore/resnet50.ckpt" \
--tag "sysu_all" \
--loss-func "id" \
--sysu_mode "all" \
--start-epoch 1 \
--epoch 80