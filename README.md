# "The Devil Is in the Word Alignment Details: On Translation-Based Cross-Lingual Transfer for Token Classification Tasks"

The repository contains the code and sample data for our paper [The Devil Is in the Word Alignment Details: On Translation-Based Cross-Lingual Transfer for Token Classification Tasks](https://arxiv.org/pdf/2505.10507).

### Dependencies

Start by installing the dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

### Running Translate-Train, Translate-Test or Ensemble-Train-Test reusing our datasets

1. Copy our final data from TBD to the [./data](data) folder.
2. Train your model using any token-classifcation training script. The original English training data can be found in [./data/original/\<task\>/train-en.jsonl](data/original/masakhaner). It is the CoNLL-2003 English data for Masakhaner2.0 (with the "MISC" entity already replaced by "O") and the Snips/Facebook training data taken from xSID. The translate-train training data can be found in [./data/final/nllb/accalign/\<task\>/train-translate-en-\<target_lang\>.jsonl](data/final/nllb/accalign/masakhaner/).
3. Run inference on the evaluation set saving the logits. We expect a single logit distribution per word (i.e., just label the first sub-word in case the word is split into multiple subwords). We provide an example test set in [./data/final/nllb/accalign/masakhaner](data/final/nllb/accalign/masakhaner) and the corresponding example logits in [./data/logits/masakhaner](data/logits/masakhaner)
4. Run one of the three scripts depending on your translation-based strategy for evaluation:

```bash
# Translate-Train
bash scripts/run_evaluation_masakhaner_ttrain.sh
# Translate-Test
bash scripts/run_evaluation_masakhaner_ttest.sh
# Ensemble-Train-Test
bash scripts/run_evaluation_masakhaner_ensemble.sh
# We provide similar scripts for xSID
```
4. The scores are saved to [./data/scores/masakhaner](data/scores/masakhaner)

### Recreating our data (or creating your own translated data)

1. Copy the source data for [xSID](https://github.com/mainlp/xsid/tree/main/data/xSID-0.5) to [./data/original/raw/xSID-0.5](data/original/raw/xSID-0.5).
2. Prepare/downloade the data original data with the following script:
```bash
# Translate-Train
bash scripts/prepare_data.sh
```
3. Translate the data using the following scripts:
```bash
# Translate-Train
bash scripts/run_translation_masakhaner_ttrain.sh
# Translate-Test
bash scripts/run_translation_masakhaner_ttest.sh
# We provide similar scripts for xSID
```
4. Prepare the data for alignment, produce the word alignments, and create the final datasets. To run [awesome-align](http://github.com/neulab/awesome-align/tree/master) please clone their repository and follow their instructions for the setup. To run AccAlign in the fine-tuned version copy their publicly released checkpoint to [./AccAlign/checkpoint-adapter](AccAlign/checkpoint-adapter)
```bash
# Acc Align without Fine-Tuning for Translate-Train
bash scripts/acc_align_no_ft_train_masakhaner.sh
# Acc Align without Fine-Tuning for Translate-Test
bash scripts/acc_align_no_ft_test_masakhaner.sh
# We provide similar scripts for xSID, and awesome-align as well as fine-tuned AccAlign
```




