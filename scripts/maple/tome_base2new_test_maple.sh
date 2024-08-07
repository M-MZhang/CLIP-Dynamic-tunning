#!/bin/bash

#cd ../..

# custom config
DATA="~/data1/zmm/data"
TRAINER=MaPLe

# DATASET=$1
# SEED=$2

DATASET=("caltech101" "oxford_pets" "stanford_cars" "oxford_flowers" "food101" "fgvc_aircraft" "sun397" "dtd" "eurosat" "ucf101")
# DATASET="caltech101"
SEED=(1 2 3)

CFG=vit_b16_c2_ep5_batch4_2ctx
SHOTS=16
LOADEP=5
SUB=new

for dataset in ${DATASET[@]}
do
    for seed in ${SEED[@]}
    do
        COMMON_DIR=${dataset}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${seed}
        MODEL_DIR=~/data1/zmm/output/base2new/train_base/${COMMON_DIR}
        OUT_DIR=${dataset}/shots_${SHOTS}/${TRAINER}_ToMe/${CFG}_8token/seed${seed}
        DIR=~/data1/zmm/output/base2new/test_${SUB}/${OUT_DIR}
        if [ -d "$DIR" ]; then
            echo "Evaluating model"
            echo "Results are available in ${DIR}. Resuming..."

            python benchmark_tome.py \
            --root ${DATA} \
            --seed ${seed} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            --model-dir ${MODEL_DIR} \
            --load-epoch ${LOADEP} \
            --eval-only \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES ${SUB}

        else
            echo "Evaluating model"
            echo "Runing the first phase job and save the output to ${DIR}"

            python benchmark_tome.py \
            --root ${DATA} \
            --seed ${seed} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${dataset}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            --model-dir ${MODEL_DIR} \
            --load-epoch ${LOADEP} \
            --eval-only \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES ${SUB}
        fi
    done
done