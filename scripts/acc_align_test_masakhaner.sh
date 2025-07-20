#!/bin/bash
WORK_DIR=/home/bee82nf/devil-in-details
TASK="masakhaner"
ORIGINAL_DATA_FILE=${WORK_DIR}/data/original/${TASK}/test-bam.jsonl
TEXT_COLUMN="tokens"
TRANSLATED_LANG="en"

# Translate all target task languages (if possible) and all sample languages
# Only high resource languages are considered
for original_lang in bam; do # bam ewe fon hau ibo kin lug luo mos nya sna swa tsn twi wol xho yor zul
    echo "Process ${original_lang}"
    OUT_DIR=${WORK_DIR}/data/intermediate/nllb/${TASK}/test-translate-${original_lang}-${TRANSLATED_LANG}
    # File containing the translations
    TRANSLATED_DATA_FILE=$OUT_DIR/test-translate-${original_lang}-${TRANSLATED_LANG}-tokens-processed.jsonl
    # File containing the input for the word alignment (original clean data)
    TRANSLATED_ALIGN_IN_FILE=$OUT_DIR/${TRANSLATED_LANG}-tokens.txt
    # File containing the input for the word alignment (translated data)
    ORIGINAL_ALIGN_IN_FILE=$OUT_DIR/${original_lang}-tokens.txt
    # File containing the alignments produced by the word aligner
    ALIGNMENT_DIR=$OUT_DIR
    # Alignment always from source to target (i.e., we project form the translated English data to the clean target language data)
    ALIGNMENT_FILE=acc-${TRANSLATED_LANG}-${original_lang}-tokens.txt
    # File for final dataset
    DATASET_FILE=${WORK_DIR}/data/final/nllb/accalign/${TASK}/test-translate-${original_lang}-${TRANSLATED_LANG}.jsonl
    echo "Prepare original and translated data for alignment"
    python $WORK_DIR/devil_in_details/alignment/prepare_alignment.py $ORIGINAL_DATA_FILE $TEXT_COLUMN ${TRANSLATED_LANG} $TRANSLATED_DATA_FILE $ORIGINAL_ALIGN_IN_FILE $TRANSLATED_ALIGN_IN_FILE --tokenizer moses
    echo "Produce word alignments"
    # We treat the translated data as source in T-Test
    bash $WORK_DIR/scripts/acc_align.sh $TRANSLATED_ALIGN_IN_FILE $ORIGINAL_ALIGN_IN_FILE $ALIGNMENT_DIR $ALIGNMENT_FILE
    echo "Postprocess word alignments"
    # We treat the translated data as source in T-Test
    python $WORK_DIR/devil_in_details/alignment/postprocess_alignment_ttest.py $TRANSLATED_ALIGN_IN_FILE $ORIGINAL_DATA_FILE $ALIGNMENT_DIR/$ALIGNMENT_FILE.6 $DATASET_FILE
done