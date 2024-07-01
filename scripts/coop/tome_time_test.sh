#!/bin/bash

#cd ../..

# custom config
DATA=~/data1/zmm/data
TRAINER=CoOp
SHOTS=16
NCTX=16
CSC=False
CTP=end

DATASET=("caltech101" "oxford_pets" "stanford_cars" "oxford_flowers" "food101" "fgvc_aircraft" "sun397" "dtd" "eurosat" "ucf101")
# DATASET="dtd"
CFG=vit_b16_ep50  # config file

for dataset in ${DATASET[@]}
do
    for SEED in 1 2 3
    do
        COMMON_DIR=${dataset}/shots_${SHOTS}/${TRAINER}/${CFG}_nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
        OUT_DIR=${dataset}/shots_${SHOTS}/${TRAINER}_ToMe/${CFG}_nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
        MODEL_DIR=~/data1/zmm/output/base2new/train_base/${COMMON_DIR}
        DIR=~/data1/zmm/output/time_test/${OUT_DIR}
        python benchmark_tome.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR}\
        --model-dir ${MODEL_DIR}\
        --load-epoch 50 \
        --eval-only \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP}\
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES new
    done
done