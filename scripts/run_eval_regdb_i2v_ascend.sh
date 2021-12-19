#!/bin/sh

myfile="run_eval_regdb_i2v_ascend.sh"

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
--dataset "RegDB" \
--data-path "Define your own path/regdb" \
--device-target Ascend \
--device-id 0 \
--resume "XXX.ckpt" \
--regdb-mode i2v \
--tag "regdb_i2v" \
--trial 1