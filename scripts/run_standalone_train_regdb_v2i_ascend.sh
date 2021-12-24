#!/bin/sh

myfile="run_standalone_train_regdb_v2i_ascend.sh"

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
--device-target Ascend \
--device-id 1 \
--pretrain "../resnet50.ckpt" \
--tag "regdb_v2i" \
--loss-func id+tri \
--trial 1 \
--regdb_mode v2i \
--epoch 80 \
--print-per-step 30