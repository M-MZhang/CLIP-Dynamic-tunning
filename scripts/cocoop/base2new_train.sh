#!/bin/bash

#cd ../..

# custom config
DATA="~/data1/zmm/data"
TRAINER=CoCoOp

DATASET=("caltech101" "oxford_pets" "stanford_cars" "oxford_flowers" "food101" "fgvc_aircraft" "sun397" "dtd" "eurosat" "ucf101")


CFG=vit_b16_c4_ep10_batch1_ctxv1
SHOTS=16

for dataset in ${DATASET[@]}
do 
    for seed in 2 3
    do 
        DIR=~/data1/zmm/output/base2new/train_base/${dataset}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${seed}
        if [ -d "$DIR" ]; then
            echo "Results are available in ${DIR}. Resuming..."
            python train.py \
            --root ${DATA} \
            --seed ${seed} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${dataset}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES base
        else
            echo "Run this job and save the output to ${DIR}"
            python train.py \
            --root ${DATA} \
            --seed ${seed} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${dataset}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES base
        fi
    done
done