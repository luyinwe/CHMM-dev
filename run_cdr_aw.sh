#!/bin/bash

# Author: Yinghao Li @ Georgia Tech ECE
# All rights reserved

# -------------------------------------
# This script is used to run txt_gen_train.py
# -------------------------------------

# Quit if there's any errors
set -e

DATASET=BC5CDR
BERT_EPOCH=100
PHASE2_EPOCH=20
BERT_BATCH_SIZE=8
DENOISING_BATCH_SIZE=64
DENOISING_EPOCH=100
DENOISING_PRETRAIN_EPOCH=5
NN_LR=0.001
MAX_SEQ_LEN=512
OUTPUT_DIR=./BC5CDR
MODEL=allenai/scibert_scivocab_uncased
SEED=3
EMISSION_NN_WEIGHT=1

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
    --obs_normalization \
    --overwrite_cache \
    --emiss_nn_weight $EMISSION_NN_WEIGHT
