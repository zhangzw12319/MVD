#!/bin/sh

myfile="run_standalone_train_regdb_v2i_gpu.sh"

if [ ! -f "$myfile" ]; then
    echo "Please first enter MVD/scripts/run_standalone_train and run. Exit..."
    exit 0
fi

cd ..

# Note: --pretrain, --data-path arguments support global path or relative path(starting
#       from project root directory, i.e. /.../DDAG_mindspore/)

python train.py \
--MSmode "GRAPH_MODE" \
--dataset RegDB \
--data-path "/home/shz/pytorch/data/regdb/" \
--optim adam \
--lr 0.00035 \
--device-target GPU \
--gpu 3 \
--pretrain "resnet50.ckpt" \
--tag "regdb_visible2infrared" \
--loss-func "id+tri" \
--trial "1" \
--regdb_mode "v2i"
