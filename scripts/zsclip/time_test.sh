# #!/bin/bash

# #cd ../..

# # custom config
# DATA=/path/to/datasets
# TRAINER=ZeroshotCLIP
# DATASET=$1
# CFG=$2  # rn50, rn101, vit_b32 or vit_b16

# python train.py \
# --root ${DATA} \
# --trainer ${TRAINER} \
# --dataset-config-file configs/datasets/${DATASET}.yaml \
# --config-file configs/trainers/CoOp/${CFG}.yaml \
# --output-dir output/${TRAINER}/${CFG}/${DATASET} \
# --eval-only

#!/bin/bash

#cd ../..
DATA="~/data1/zmm/data"
TRAINER=ZeroshotCLIP

DATASET=("caltech101" "oxford_pets" "stanford_cars" "oxford_flowers" "food101" "fgvc_aircraft" "sun397" "dtd" "eurosat" "ucf101")
# DATASET="ucf101"
SEED=(1 2 3)

CFG=vit_b16
SHOTS=16
LOADEP=5
SUB=new

for dataset in ${DATASET[@]}
do
    for seed in ${SEED[@]}
    do
        COMMON_DIR=${dataset}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${seed}
        DIR=~/data1/zmm/output/time_test/${COMMON_DIR}
        if [ -d "$DIR" ]; then
            echo "Evaluating model"
            echo "Results are available in ${DIR}. Resuming..."

            python benchmark.py \
            --root ${DATA} \
            --seed ${seed} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            --eval-only \
            DATASET.NUM_SHOTS ${SHOTS} \

        else
            echo "Evaluating model"
            echo "Runing the first phase job and save the output to ${DIR}"

            python benchmark.py \
            --root ${DATA} \
            --seed ${seed} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${dataset}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            --eval-only \
            DATASET.NUM_SHOTS ${SHOTS} 
        fi
    done
done