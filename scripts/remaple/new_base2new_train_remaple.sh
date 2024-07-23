#!/bin/bash

#cd ../..

# custom config
DATA="~/data1/zmm/data"
TRAINER=ReMaPLe

# DATASET=$1
# SEED=$2

DATASET=("caltech101" "oxford_pets" "stanford_cars" "oxford_flowers" "food101" "fgvc_aircraft" "sun397" "dtd" "eurosat" "ucf101")
# DATASET="sun397"
SEED=(1 2 3)

CFG=vit_b16_c2_ep5_batch4_2ctx
SHOTS=16

for dataset in ${DATASET[@]}
do
    for seed in ${SEED[@]}
    do
        DIR=~/data1/zmm/output/base2new/train_base/${dataset}/shots_${SHOTS}/${TRAINER}_3_1/${CFG}/seed${seed}
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