#!/bin/bash
WORK_DIR=$PWD

TASK='xsid'
SPLIT='test'
TRG_LANG='en'
MODEL='facebook/nllb-200-3.3B'
BATCH_SIZE=8

for SRC_LANG in ar da de-st de id it kk nl sr tr zh; do
    # Path with the downloaded source data
    SRC_PATH=$WORK_DIR/data/original/${TASK}/${SPLIT}-${SRC_LANG}.jsonl
    OUT_PATH=$WORK_DIR/data/intermediate/nllb/${TASK}/${SPLIT}-translate-${SRC_LANG}-${TRG_LANG}/${SPLIT}-translate-${SRC_LANG}-${TRG_LANG}-tokens.jsonl
    for COLUMN in tokens; do
        # Path to save the preprocessed data to
        SRC_PATH_PRE=$WORK_DIR/data/original/${TASK}/${SPLIT}-${SRC_LANG}/${SPLIT}-${SRC_LANG}-${COLUMN}.jsonl
        ### Translate
        bash $WORK_DIR/scripts/run_translation.sh ${MODEL} ${BATCH_SIZE} ${SRC_PATH} ${SRC_PATH_PRE} ${OUT_PATH} ${SRC_LANG} ${TRG_LANG} ${COLUMN}
    done
done