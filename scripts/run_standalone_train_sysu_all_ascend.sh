#!/bin/sh

myfile="run_standalone_train_sysu_all_ascend.sh"

if [ ! -f "$myfile" ]; then
    echo "Please first enter MVD/scripts/run_standalone_train and run. Exit..."
    exit 0
fi

cd ..

# Note: --pretrain, --data-path arguments support global path or relative path(starting
#       from project root directory, i.e. /.../DDAG_mindspore/)

python train.py \
--MSmode PYNATIVE_MODE \
--dataset SYSU \
--data-path "Define your own path/sysu/" \
--optim adam \
--lr 0.0035 \
--device-target Ascend \
--device-id 0 \
--pretrain "resnet50.ckpt" \
--tag "sysu_all" \
--loss-func id+tri \
--sysu-mode all \
--epoch 80 \
--print-per-step 100