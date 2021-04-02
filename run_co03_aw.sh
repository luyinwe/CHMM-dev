#!/bin/bash

# Author: Yinghao Li @ Georgia Tech ECE
# All rights reserved

# -------------------------------------
# This script is used to run txt_gen_train.py
# -------------------------------------

# Quit if there's any errors
set -e

DATASET=Co03
BERT_EPOCH=15
PHASE2_EPOCH=5
BERT_BATCH_SIZE=16
DENOISING_BATCH_SIZE=32
DENOISING_EPOCH=10
DENOISING_PRETRAIN_EPOCH=5
NN_LR=0.00001
MAX_SEQ_LEN=256
OUTPUT_DIR=./Co03
MODEL=bert-base-uncased
SEED=1
EMISSION_NN_WEIGHT=0

CUDA_VISIBLE_DEVICES=$1 python alt_train.py \
    --data_dir ../data/ \
    --output_dir $OUTPUT_DIR \
    --dataset_name $DATASET \
    --denoising_model nhmm \
    --denoising_batch_size $DENOISING_BATCH_SIZE \
    --denoising_epoch $DENOISING_EPOCH \
    --denoising_pretrain_epoch $DENOISING_PRETRAIN_EPOCH \
    --nn_lr $NN_LR \
    --seed $SEED \
    --overwrite_cache \
    --converse_first \
    --emiss_nn_weight $EMISSION_NN_WEIGHT \
    --use_src_attention_weights
