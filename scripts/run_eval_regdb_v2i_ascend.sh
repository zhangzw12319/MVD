#!/bin/sh

myfile="run_eval_regdb_v2i_ascend.sh"

if [ ! -f "$myfile" ]; then
    echo "Please first enter MVD/scripts/ and run. Exit..."
    exit 0
fi

cd ..

# Note: --resume, --data-path arguments support global path or relative path(starting
#       from project root directory, i.e. /.../DDAG_mindspore/)
# if --device-target GPU, then set --gpu X
# if --device-target Ascend, then set --device-id X

python eval.py \
--MSmode GRAPH_MODE \
--dataset RegDB \
--data-path "../dataset/regdb" \
--device-target Ascend \
--device-id 0 \
--resume "epoch_20_rank1_77.33_mAP_73.12_RegDB_batch-size_2*8*4=64_adam_lr_0.0035_loss-func_id+tri_trial_1_main.ckpt" \
--regdb-mode v2i \
--tag "regdb_v2i" \
--trial 1