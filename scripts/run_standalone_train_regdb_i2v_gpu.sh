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
--MSmode GRAPH_MODE \
--dataset RegDB \
--data-path "Define your own path/regdb/" \
--optim adam \
--lr 0.0035 \
--device-target GPU \
--gpu 0 \
--pretrain "resnet50.ckpt" \
--tag "regdb_i2v" \
--loss-func id+tri \
--trial 1 \
--regdb_mode i2v \
--epoch 80 \
--print-per-step 30