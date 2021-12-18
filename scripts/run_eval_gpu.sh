#!/bin/sh

myfile="run_eval.sh"

if [ ! -f "$myfile" ]; then
    echo "Please first enter MVD/scripts/ and run. Exit..."
    exit 0
fi

cd ..

# Note: --resume, --data-path arguments support global path or relative path(starting
#       from project root directory, i.e. /.../DDAG_mindspore/)

python eval.py \
--dataset "SYSU" \
--data-path "/home/shz/pytorch/data/sysu" \
--device-target GPU \
--gpu 2 \
--resume "logs/sysu_all/training/epoch_20_rank1_58.05_mAP_56.17_SYSU_batch-size_2*8*4=64_adam_lr_0.00035_loss-func_id+tri_main.ckpt" \
--tag "sysu_all"