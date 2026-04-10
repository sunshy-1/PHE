#!/bin/bash

# ============ Common Variables ============
PRETRAIN_MODEL_DIR="pretrain_models/phe_hgt_CS"
DATA_DIR="./OAG_dataset"
N_EPOCH_PRETRAIN=500
N_EPOCH_FINETUNE=100
CUDA=6

# # ============ Pretrain ============
# python pretrain_OAG_PHE.py \
#     --data_dir "${DATA_DIR}/graph_CS.pk" \
#     --pretrain_model_dir "${PRETRAIN_MODEL_DIR}" \
#     --n_epoch ${N_EPOCH_PRETRAIN} \
#     --cuda ${CUDA} \
#     --batch_size 2048 \
#     --conv_name 'hgt'

# ============ Finetune ============
DOMAINS=("_Engineering" "_CS" "_Materials")
TASKS=("AD" "PV" "PF")

for domain in "${DOMAINS[@]}"; do
    for task in "${TASKS[@]}"; do
        echo "Running finetune_${task}.py with domain=${domain}"
        python "finetune_${task}.py" \
            --pretrain_model_dir "${PRETRAIN_MODEL_DIR}" \
            --data_dir "${DATA_DIR}" \
            --domain "${domain}" \
            --n_epoch ${N_EPOCH_FINETUNE} \
            --cuda ${CUDA}
    done
done
