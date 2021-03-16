#!/bin/bash

# Author: Yinghao Li @ Georgia Tech ECE
# All rights reserved

# -------------------------------------
# This script is used to run txt_gen_train.py
# -------------------------------------

# Quit if there's any errors
set -e

DATASET=BC5CDR
TASK=source

CUDA_VISIBLE_DEVICES=1 python universal_test.py \
  --dataset "$DATASET" \
  --test_task "$TASK"
