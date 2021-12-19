#!/bin/sh

myfile="run_eval_sysu_indoor_ascend.sh"

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
--MSmode "PYNATIVE_MODE"
--dataset "SYSU" \
--data-path "Define your own path/sysu" \
--device-target Ascend \
--device-id 0 \
--resume "XXX.ckpt" \
--sysu-mode indoor \
--tag "sysu_indoor"