#!/bin/sh

myfile="run_eval_sysu_all_gpu.sh"

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
--MSmode "PYNATIVE_MODE" \
--dataset "SYSU" \
--data-path "Define your own path/sysu" \
--device-target GPU \
--gpu 0 \
--resume "../epoch_20.ckpt" \
--sysu-mode all \
--tag "sysu_all"