#!/bin/bash
WORK_DIR=/home/bee82nf/devil-in-details
TASK="xsid"
ORIGINAL_DATA_FILE=${WORK_DIR}/data/original/${TASK}/test-bam.jsonl
TEXT_COLUMN="tokens"
TRANSLATED_LANG="en"

# Translate all target task languages (if possible) and all sample languages
# Only high resource languages are considered
for original_lang in ar da de de-st id it kk nl sr tr zh; do
    echo "Process ${original_lang}"
    OUT_DIR=${WORK_DIR}/data/intermediate/nllb/${TASK}/test-translate-${original_lang}-${TRANSLATED_LANG}
    # File containing the translations
    TRANSLATED_DATA_FILE=$OUT_DIR/test-translate-${original_lang}-en-tokens-processed.jsonl
    # File containing the input for the word alignment (original clean data)
    TRANSLATED_ALIGN_IN_FILE=$OUT_DIR/${TRANSLATED_LANG}-tokens.txt
    # File containing the input for the word alignment (translated data)
    ORIGINAL_ALIGN_IN_FILE=$OUT_DIR/${original_lang}-tokens.txt
    # File containing the merged data
    MERGED_FILE=$OUT_DIR/${TRANSLATED_LANG}-${original_lang}-tokens.txt
    # Alignment always from source to target (i.e., we project form the translated English data to the clean target language data)
    ALIGNMENT_FILE=$OUT_DIR/awesome-${TRANSLATED_LANG}-${original_lang}-tokens.txt
    # File for final dataset
    DATASET_FILE=${WORK_DIR}/data/final/nllb/awesomealign/${TASK}/test-translate-${original_lang}-${TRANSLATED_LANG}.jsonl
    echo "Prepare original and translated data for alignment"
    python $WORK_DIR/devil_in_details/alignment/prepare_alignment.py $ORIGINAL_DATA_FILE $TEXT_COLUMN ${TRANSLATED_LANG} $TRANSLATED_DATA_FILE $ORIGINAL_ALIGN_IN_FILE $TRANSLATED_ALIGN_IN_FILE --tokenizer moses
    echo "Produce word alignments"
    # We treat the translated data as source in T-Test
    bash $WORK_DIR/scripts/awesome_align.sh $TRANSLATED_ALIGN_IN_FILE $ORIGINAL_ALIGN_IN_FILE $MERGED_FILE $ALIGNMENT_FILE
    echo "Postprocess word alignments"
    # We treat the translated data as source in T-Test
    python $WORK_DIR/devil_in_details/alignment/postprocess_alignment_ttest.py $TRANSLATED_ALIGN_IN_FILE $ORIGINAL_DATA_FILE $ALIGNMENT_FILE $DATASET_FILE
done