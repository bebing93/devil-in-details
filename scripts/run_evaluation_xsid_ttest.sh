#!/bin/bash

# TODO: Adjust to your needs
WORK_DIR=$PWD
SOURCE_LANG="en"
TASK="xsid"
EVAL_SETTING="ttest"
SPLIT=test

# Word aligner used to create the alignments for the translated data
WORD_ALIGNER=accalign

# Tokenizer of the downstream model (used for sanity checking)
TOKENIZER="FacebookAI/xlm-roberta-large"


base_data_dir="$WORK_DIR/data/final/nllb/${WORD_ALIGNER}/${TASK}"
base_logits_dir="$WORK_DIR/data/logits/${TASK}/${EVAL_SETTING}"
base_scores_dir="$WORK_DIR/data/scores/${TASK}/${EVAL_SETTING}"

for target_lang in ar da de de-st id it kk nl sr tr zh; do
    dataset_path="${base_data_dir}/${SPLIT}-translate-${target_lang}-${SOURCE_LANG}.jsonl"
    # Project translate test logits
    logit_file="${SPLIT}_${target_lang}_logits.pt"
    projected_logit_file="${SPLIT}_${target_lang}_projected_logits.pt"
    python $WORK_DIR/devil_in_details/evaluation/project_translate_test_logits_bio.py ${dataset_path} ${base_logits_dir}/${logit_file} ${base_logits_dir}/${projected_logit_file} --tokenizer_path ${TOKENIZER} --restrict_target

    # Compute the score for every target lang
    out_score_file="${target_lang}_score.txt"
    # HINT: Mapping from int to label string depends on task
    python $WORK_DIR/devil_in_details/evaluation/evaluate_bio.py ${TASK} ${dataset_path} ${base_scores_dir}/${out_score_file} ${base_logits_dir}/${projected_logit_file}
done