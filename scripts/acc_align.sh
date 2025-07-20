#!/bin/sh
SOURCE_FILE=$1
TARGET_FILE=$2
OUTPUT_DIR=$3
OUTPUT_FILE=$4
ALIGNMENT_THRESHOLD=${5:-0.1}
BATCH_SIZE=${6:-32}

ADAPTER=/data/42-julia-hpc-rz-wuenlp/bee82nf/.cache/huggingface/adapter/checkpoint
MODEL='sentence-transformers/LaBSE'

python AccAlign/train_alignment_adapter.py \
    --infer_path $OUTPUT_DIR \
    --infer_filename $OUTPUT_FILE \
    --adapter_path $ADAPTER \
    --model_name_or_path $MODEL \
    --extraction 'softmax' \
    --infer_data_file_src $SOURCE_FILE \
    --infer_data_file_tgt $TARGET_FILE \
    --per_gpu_train_batch_size ${BATCH_SIZE} \
    --align_layer 6 \
    --softmax_threshold $ALIGNMENT_THRESHOLD \
    --do_test \

exit