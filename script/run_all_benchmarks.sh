#!/bin/bash
# """
# run all benchmark in CPFT
# """

export CUDA_VISIBLE_DEVICES=1

for data in HWU64 BANKING77 CLINC150
do
    python fsid_main.py \
        --data_name $data \
        --n_shot 5

        
    python fsid_main.py \
        --data_name $data \
        --ckpt_dir /home/keonwookim/something-FSID/SimCSE/result/cpft-unsup-simcse-roberta-base \
        --n_shot 5
done
