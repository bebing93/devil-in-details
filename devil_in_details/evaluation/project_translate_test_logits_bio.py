import os
import sys
import argparse
import logging
from typing import List, Dict, Optional

import torch
import datasets
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from devil_in_details.utils import (
    extract_entity_indices,
    load_logits_with_retry,
    build_alignment_mapping,
)


def map_entities(
    source_indices: List[List[int]],
    srcid2trgid: Dict[int, List[int]],
    complete_source: bool,  # Source of the mapping
    complete_target: bool,  # Target of the mapping
) -> Optional[List[int]]:
    """Map source entities (translated data) to target entities (clean data) using alignment.

    Args:
        source_indices: Indices of the source entity
        srcid2trgid: Source-to-target alignment mapping
        complete_source: Checks whether the source mapping is complete
        complete_target: Checks whether the target mapping is complete

    Returns:
        List of target indices
    """

    target_indices = []
    # Collect the corresponding translated indices
    for src_idx in source_indices:
        if src_idx in srcid2trgid:
            target_indices = target_indices + srcid2trgid[src_idx]
        elif complete_source:  # COMP-SRC
            # We couldn´t map the source mapping entity completely
            return None

    if not target_indices:
        return None  # Nothing to map

    # Build the span of indices
    target_indices = sorted(set(target_indices))
    min_idx, max_idx = target_indices[0], target_indices[-1]

    # Check that the entity covers a consecutive number of indices (COMP-TGT)
    if complete_target:  # COMP-TGT
        expected_indices = list(range(min_idx, max_idx + 1))
        if target_indices != expected_indices:
            # The target mapping entity is incomplete
            return None

    return target_indices


def project_translate_test_logits_bio(
    target_data_path,
    source_logit_path,
    target_logit_path,
    text_column="tokens",
    label_column="org_ner_tags",
    alignment_column="alignment",
    num_labels=7,
    tokenizer_path="xlm-roberta-large",
    max_length=256,
    complete_source=False,
    complete_target=False,
    restrict_target=True,
    **kwargs,
):
    logger.info(f"Loading dataset from {target_data_path}")
    # Load data
    test_data = datasets.load_dataset(
        path=f"json",
        data_files=target_data_path,
        split="train",
    )

    # Load logits
    logger.info(f"Loading logits from {source_logit_path}")
    all_source_logits = load_logits_with_retry(source_logit_path)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Santiy check: As many predictions as we have examples
    if len(test_data[label_column]) != len(all_source_logits):
        raise ValueError(
            f"Data size mismatch: {len(test_data[label_column])} examples vs "
            f"{len(all_source_logits)} logit sets"
        )

    # Projecting logits
    total_entities = 0
    all_target_logits = []
    for example, source_logits in zip(test_data, all_source_logits):
        src_tokens = example[text_column]

        # Sanity check: as many logits per example as we have tokens (get the label lenght produced by the downstream tokenizer)
        word_ids = tokenizer(
            src_tokens,
            max_length=max_length,
            truncation=True,
            is_split_into_words=True,
        ).word_ids(batch_index=0)
        word_ids = [word_idx for word_idx in word_ids if word_idx != None]

        expected_length = len(src_tokens[: max(word_ids) + 1])
        if len(source_logits) != expected_length:
            raise ValueError(
                f"Logits length ({len(source_logits)}) doesn't match expected length ({expected_length})"
            )

        # Clean target data labels for evaluation
        labels = example[label_column]

        # Create place holder for projected logits
        zero_preds = torch.zeros((num_labels))
        # Giving extremly high logits to label 0 ==> Make it possible to detect those for ensembling
        zero_preds[0] = sys.maxsize
        target_logits = [zero_preds] * len(labels)

        # Create mapping lookup from trans token id to list of src token ids
        alignment = example[alignment_column]
        srcid2trgid = build_alignment_mapping(alignment, inverse=False)

        # Get the predictions on the translated data
        source_predictions = [token.argmax().item() for token in source_logits]

        # Get the entity´s indices of the unprojected predicitons
        # From: [0,0,1,2,0,...]
        # To: [[2,3],...]
        source_entities, _ = extract_entity_indices(source_predictions)

        # Sanity check that we found all entities
        flattened_entities = sum(source_entities, [])
        reconstructed = [
            source_predictions[idx] if idx in flattened_entities else 0
            for idx in range(len(source_predictions))
        ]
        if source_predictions != reconstructed:
            logger.warning(
                "Entity extraction validation failed - some entities may be missed"
            )

        # Project each entity (logits from translated data)
        for entity_indices in source_entities:

            target_indices = map_entities(
                entity_indices,
                srcid2trgid,
                complete_source,
                complete_target,
            )

            if target_indices is None:
                continue

            min_idx, max_idx = min(target_indices), max(target_indices)

            target_logits[min_idx] = source_logits[entity_indices[0]]

            # Check the filters
            if restrict_target:
                if len(entity_indices) == 1:
                    continue

            # We have multiple target mappings
            if min_idx != max_idx:
                if len(entity_indices) == 1:
                    # We have a single source token, but multiple target tokens
                    current_tag = source_logits[entity_indices[0]].argmax().item()
                    if current_tag % 2 == 1:
                        # We have a B-Tag at the beginnig, we need to artifically create an I-Tag
                        i_tag_logit = torch.zeros_like(source_logits[entity_indices[0]])
                        # Exchange logits for B and I tag aka replace the B- with an I-Tag
                        i_tag_logit[current_tag + 1], i_tag_logit[current_tag] = (
                            source_logits[entity_indices[0]][current_tag],
                            source_logits[entity_indices[0]][current_tag + 1],
                        )
                    else:
                        # We have an I-Tag at the beginning, we can use it
                        i_tag_logit = source_logits[entity_indices[0]]
                else:
                    # We have multple source tokens and multiple target tokens, take the last logit from the predictions
                    i_tag_logit = source_logits[entity_indices[-1]]

                for jj in range(min_idx + 1, max_idx + 1):
                    target_logits[jj] = i_tag_logit

        # Sanity check
        if len(target_logits) != len(labels):
            raise ValueError("Lenght of projected tokens doesn't match input data")

        all_target_logits.append(target_logits)

        # Count projected entities
        target_preds = [token.argmax().item() for token in target_logits]
        entities = extract_entity_indices(target_preds)
        total_entities += len(entities)

    # Sanity check
    if len(all_target_logits) != len(test_data[label_column]):
        raise ValueError("Number of projected instances doesn't match input data")

    logger.info(f"Projected {total_entities} entities total")

    # Save results
    logger.info(f"Saving projected logits to {target_logit_path}")
    os.makedirs(os.path.dirname(target_logit_path), exist_ok=True)
    torch.save(all_target_logits, target_logit_path)
    logger.info("Projection complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target_data_path",
        help="Path to the clean evaluation data in the target language",
    )
    parser.add_argument(
        "source_logit_path",
        help="Path to the predicted logits for the translated source language data",
    )
    parser.add_argument(
        "target_logit_path",
        help="Path to save the projected logits for the clean target language data",
    )
    parser.add_argument(
        "--text_column", default="tokens", help="Column with the input data"
    )
    parser.add_argument(
        "--label_column",
        default="org_ner_tags",
        help="Column with the clean target language labels",
    )
    parser.add_argument(
        "--alignment_column",
        default="alignment",
        help="Column containing the alignments from source to target",
    )
    parser.add_argument("--num_labels", default=7, type=int, help="Number of labels")
    parser.add_argument(
        "--tokenizer_path",
        default="xlm-roberta-large",
        help="Tokenizer used for the downstream model (for sanity checking)",
    )
    parser.add_argument(
        "--complete_source",
        action="store_true",
        help="Require complete target entity alignment",
    )
    parser.add_argument(
        "--complete_target",
        action="store_true",
        help="Require complete target entity alignment",
    )
    parser.add_argument(
        "--restrict_target",
        action="store_true",
        help="Restricted target entity alignment",
    )
    args = parser.parse_args()

    project_translate_test_logits_bio(
        args.target_data_path,
        args.source_logit_path,
        args.target_logit_path,
        args.text_column,
        args.label_column,
        args.alignment_column,
        args.num_labels,
        args.tokenizer_path,
        complete_source=args.complete_source,
        complete_target=args.complete_target,
        restrict_target=args.restrict_target,
    )
