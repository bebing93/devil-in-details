#!/bin/bash
WORK_DIR=$PWD
TASK="masakhaner"
ORIGINAL_DATA_FILE=${WORK_DIR}/data/original/${TASK}/train-en.jsonl
TEXT_COLUMN="tokens"
ORIGINAL_LANG="en"
ALIGNER=awesomealign_noft

for translated_lang in bam ewe fon hau ibo kin lug luo mos nya sna swa tsn twi wol xho yor zul; do # ewe fon hau ibo kin lug luo mos nya sna swa tsn twi wol xho yor zul
    echo "Process ${translated_lang}"
    OUT_DIR=${WORK_DIR}/data/intermediate/nllb/${TASK}/train-translate-${ORIGINAL_LANG}-${translated_lang}
    # File containing the translations
    TRANSLATED_DATA_FILE=$OUT_DIR/train-translate-${ORIGINAL_LANG}-${translated_lang}-tokens-processed.jsonl
    # File containing the input for the word alignment (original clean data)
    ORIGINAL_ALIGN_IN_FILE=$OUT_DIR/${ORIGINAL_LANG}-tokens.txt
    # File containing the input for the word alignment (translated data)
    TRANSLATED_ALIGN_IN_FILE=$OUT_DIR/${translated_lang}-tokens.txt
    # File containing both the original and translated data separated by ||| 
    MERGED_FILE=$OUT_DIR/${ORIGINAL_LANG}-${translated_lang}-tokens.txt
    # File containing the alignments produced by the word aligner
    ALIGNMENT_FILE=$OUT_DIR/awesome_noft-${ORIGINAL_LANG}-${translated_lang}-tokens.txt
    # File for final dataset
    DATASET_FILE=${WORK_DIR}/data/final/nllb/${ALIGNER}/${TASK}/train-translate-${ORIGINAL_LANG}-${translated_lang}.jsonl
    echo "Prepare original and translated data for alignment"
    python $WORK_DIR/devil_in_details/alignment/prepare_alignment.py $ORIGINAL_DATA_FILE $TEXT_COLUMN $translated_lang $TRANSLATED_DATA_FILE $ORIGINAL_ALIGN_IN_FILE $TRANSLATED_ALIGN_IN_FILE
    echo "Produce word alignments"
    bash $WORK_DIR/scripts/awesome_align.sh $ORIGINAL_ALIGN_IN_FILE $TRANSLATED_ALIGN_IN_FILE $MERGED_FILE $ALIGNMENT_FILE
    echo "Postprocess word alignments"
    python $WORK_DIR/devil_in_details/alignment/postprocess_alignment_ttrain.py $ORIGINAL_DATA_FILE $TRANSLATED_ALIGN_IN_FILE $ALIGNMENT_FILE $DATASET_FILE --complete_source --complete_target --complete_instance
done