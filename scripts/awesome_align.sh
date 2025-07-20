#!/bin/bash
SOURCE_FILE=$1
TARGET_FILE=$2
MERGED_FILE=$3 # Combines source and target file into a joint file; each line separted by |||
OUTPUT_FILE=$4
ALIGNMENT_THRESHOLD=${5:-0.001}
BATCH_SIZE=${6:-32}

awk 'NR==FNR {a[NR]=$0; next} {print a[FNR] " ||| " $0}' ${SOURCE_FILE} ${TARGET_FILE} > ${MERGED_FILE}

MODEL_NAME_OR_PATH=bert-base-multilingual-cased

CUDA_VISIBLE_DEVICES=0 awesome-align \
    --output_file ${OUTPUT_FILE} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_file ${MERGED_FILE} \
    --extraction 'softmax' \
    --batch_size $BATCH_SIZE \
    --softmax_threshold $ALIGNMENT_THRESHOLD\