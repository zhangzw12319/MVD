#!/bin/sh

myfile="run_standalone_train_regdb_i2v_gpu.sh"

if [ ! -f "$myfile" ]; then
    echo "Please first enter MVD/scripts/ and run. Exit..."
    exit 0
fi

cd ..

# Note: --pretrain, --data-path arguments support global path or relative path(starting
#       from project root directory, i.e. /.../DDAG_mindspore/)

python train.py \
--dataset RegDB \
--data-path "/home/shz/pytorch/data/regdb/" \
--optim adam \
--lr 0.00035 \
--device-target GPU \
--gpu 2 \
--pretrain "resnet50.ckpt" \
--tag "regdb_infrared2visible" \
--loss-func "id" \
--trial "1" \
--regdb_mode "i2v" \
--MSmode "PYNATIVE_MODE"
