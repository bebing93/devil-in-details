#!/bin/bash
WORK_DIR=$PWD

MODEL=${1:-"facebook/nllb-200-3.3B"}
BATCH_SIZE=${2:-2}
SRC_PATH=$3 # Path to the original data
SRC_PATH_PRE=$4 # Path to the preprocessed original data
OUT_PATH=$5 # Output path for the translated data
SRC_LANG=$6
TRG_LANG=$7
COLUMN=$8 # Text column that is translated

SEED=20230110

echo Source Path: $SRC_PATH
echo Source Path Preprocessed: $SRC_PATH_PRE
echo Output Path: $OUT_PATH
echo Model: $MODEL

### Preprocessing
python $WORK_DIR/devil_in_details/translation/preprocess_translation.py ${SRC_PATH} ${SRC_PATH_PRE} ${COLUMN} ${SRC_LANG}

# Translation
python $WORK_DIR/devil_in_details/translation/run_translation.py ${SRC_PATH_PRE} ${OUT_PATH} --src_lang ${SRC_LANG} --trg_lang ${TRG_LANG} --model ${MODEL} --batch_size ${BATCH_SIZE} --device "cuda"

### Postprocessing
python $WORK_DIR/devil_in_details/translation/postprocess_translation.py ${OUT_PATH} ${TRG_LANG}