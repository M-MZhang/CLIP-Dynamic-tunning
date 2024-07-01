#!/bin/bash

#cd ../..

# custom config
DATA="~/data1/zmm/data"
TRAINER=CoOp

DATASET=("caltech101" "oxford_pets" "stanford_cars" "oxford_flowers" "food101" "fgvc_aircraft" "sun397" "dtd" "eurosat" "ucf101")
# DATASET="caltech101"
CFG=vit_b16_ep50  # config file
CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)

for dataset in ${DATASET[@]}
do
    for SEED in 1 2 3 
    do
        DIR=~/data1/zmm/output/base2new/train_base/${dataset}/shots_${SHOTS}/${TRAINER}/${CFG}_nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
        if [ -d "$DIR" ]; then
            echo "Results are available in ${DIR}. Skip this job"
        else
            echo "Run this job and save the output to ${DIR}"
            python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${dataset}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            TRAINER.COOP.N_CTX ${NCTX} \
            TRAINER.COOP.CSC ${CSC} \
            TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES base
        fi
    done
done