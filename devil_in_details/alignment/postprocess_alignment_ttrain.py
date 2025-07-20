import argparse
import logging
from typing import List, Dict, Optional, Any
from devil_in_details.utils import (
    load_jsonl,
    load_text_lines,
    extract_entity_indices,
    save_jsonl,
    parse_alignment_line,
    build_alignment_mapping,
)

from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def map_entities(
    src_entity_indices: List[List[int]],
    src_entity_types: List[int],
    srcidx2trgidx: Dict[int, List[int]],
    src_tokens: List[str],
    trg_tokens: List[str],
    complete_source: bool,
    complete_target: bool,
) -> Optional[Dict[int, Dict[str, Any]]]:
    """Map source entities to target entities using alignment.

    Args:
        src_entity_indices: List of source entity token indices
        src_entity_types: List of source entity types
        alignment_mapping: source-to-target alignment mapping
        source_tokens: source language tokens
        target_tokens: target language tokens

    Returns:
        Dictionary mapping entity numbers to entity information, or None if mapping fails
    """
    entity_mappings = {}

    for entity_idx, src_indices in enumerate(src_entity_indices):
        trg_indices = []

        # Collect the corresponding target indices
        for src_idx in src_indices:
            if src_idx in srcidx2trgidx:
                trg_indices = trg_indices + srcidx2trgidx[src_idx]
            elif complete_source:  # COMP-SRC
                # We couldn´t map the source entity completely
                return None

        if not trg_indices:
            continue  # Nothing to map

        # Build the span of indices
        trg_indices = sorted(set(trg_indices))
        min_idx, max_idx = trg_indices[0], trg_indices[-1]
        expected_indices = list(range(min_idx, max_idx + 1))

        # Check that the entity covers a consecutive number of indices (COMP-TGT)
        if complete_target:  # COMP-TGT
            if trg_indices != expected_indices:
                # The entity in the target language not complete
                return None

        entity_mappings[entity_idx] = {
            "src_tokens": [src_tokens[i] for i in src_indices],
            "src_indices": src_indices,
            "src_type": src_entity_types[entity_idx],
            "trg_tokens": [trg_tokens[i] for i in expected_indices],
            "trg_indices": expected_indices,
        }

    return entity_mappings


def create_target_tags(
    entity_mappings: Dict[int, Dict[str, Any]], trg_length: int
) -> List[int]:
    """Create BIO tags for target sequence.

    Args:
        entity_mappings: Mapped entities
        trg_length: Length of target sequence

    Returns:
        List of BIO tags for target sequence
    """
    trg_tags = [0] * trg_length

    for mapping in entity_mappings.values():
        trg_indices = mapping["trg_indices"]
        src_type = mapping["src_type"]

        if trg_indices:
            # First token gets B tag (same as source)
            trg_tags[trg_indices[0]] = src_type
            # Subsequent tokens get I tag
            for idx in trg_indices[1:]:
                trg_tags[idx] = src_type + 1

    return trg_tags


def validate_instance_completeness(
    src_entity_types: List[int], trg_tags: List[int]
) -> bool:
    """Validate that source and target have same entity type counts.

    Args:
        src_entity_types: source entity types
        trg_tags: target BIO tags

    Returns:
        True if counts match, False otherwise
    """
    src_tags_count = Counter(src_entity_types)
    trg_tags_count = Counter(trg_tags)

    for entity_type in set(src_entity_types):
        if src_tags_count[entity_type] != trg_tags_count[entity_type]:
            return False

    return True


def postprocess_bio_alignment(
    source_file: str,
    target_file: str,
    alignment_file: str,
    out_file: str,
    text_column: str = "tokens",
    tag_name: str = "ner_tags",
    complete_source: bool = False,
    complete_target: bool = False,
    complete_instance: bool = False,
):

    logger.info("Loading data files...")
    # Read data
    source_data = load_jsonl(source_file)
    target_data = load_text_lines(target_file)
    alignment_out_data = load_text_lines(alignment_file)

    # Validate data consistency
    if not (len(source_data) == len(target_data) == len(alignment_out_data)):
        raise ValueError(
            f"Data length mismatch: source={len(source_data)}, "
            f"target={len(target_data)}, alignment={len(alignment_out_data)}"
        )

    corrupted_indices = []
    processed_data = []
    for idx, (src_line, trg_line, alignment_line) in enumerate(
        zip(source_data, target_data, alignment_out_data)
    ):
        try:
            # Parse inputs
            src_tokens = src_line[text_column]
            trg_tokens = trg_line.split()
            alignment_pairs = parse_alignment_line(alignment_line)

            # Extract source entities
            src_entity_indices, src_entity_types = extract_entity_indices(
                src_line[tag_name]
            )

            # Build source to target mapping
            srcidx2trgidx = build_alignment_mapping(alignment_pairs)

            # Map entities
            entity_mappings = map_entities(
                src_entity_indices,
                src_entity_types,
                srcidx2trgidx,
                src_tokens,
                trg_tokens,
                complete_source,
                complete_target,
            )

            if entity_mappings is None:
                corrupted_indices.append(idx)
                continue

            # Create target tags
            trg_tags = create_target_tags(entity_mappings, len(trg_tokens))

            # Validate instance completeness (COMP-INS)
            if complete_instance:
                if not validate_instance_completeness(src_entity_types, trg_tags):
                    corrupted_indices.append(idx)
                    continue

            # Create output item
            output_item = src_line.copy()
            output_item[f"org_{text_column}"] = src_tokens
            output_item[text_column] = trg_tokens
            output_item[f"org_{tag_name}"] = src_line[tag_name]
            output_item[tag_name] = trg_tags

            processed_data.append(output_item)

        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            corrupted_indices.append(idx)

    # Report results
    total_items = len(source_data)
    corrupted_count = len(corrupted_indices)
    recovery_rate = (total_items - corrupted_count) / total_items

    logger.info(f"Processing complete:")
    logger.info(f"  Total items: {total_items}")
    logger.info(f"  Corrupted items: {corrupted_count}")
    logger.info(f"  Recovery rate: {recovery_rate:.2%}")

    # Save results
    logger.info(f"Saving results to {out_file}")
    save_jsonl(processed_data, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Align BIO tags between source and target using word alignments"
    )
    parser.add_argument(
        "source_file",
        help="Source data file in JSONL format (usually clean English data)",
    )
    parser.add_argument(
        "target_file",
        help="Target data file containing the translated target language instances",
    )
    parser.add_argument(
        "alignment_file", help="Word alignment file mapping from source to target"
    )
    parser.add_argument("out_file", help="Output file path (JSONL format)")
    parser.add_argument(
        "--text_column", default="tokens", help="Name of tokens column in source file"
    )
    parser.add_argument(
        "--tag_name", default="ner_tags", help="Name of label column in source file"
    )
    parser.add_argument(
        "--complete_source",
        action="store_true",
        help="Require complete source entity alignment",
    )
    parser.add_argument(
        "--complete_target",
        action="store_true",
        help="Require complete target entity alignment",
    )
    parser.add_argument(
        "--complete_instance",
        action="store_true",
        help="Require complete instance alignment",
    )

    args = parser.parse_args()

    postprocess_bio_alignment(
        source_file=args.source_file,
        target_file=args.target_file,
        alignment_file=args.alignment_file,
        out_file=args.out_file,
        text_column=args.text_column,
        tag_name=args.tag_name,
        complete_source=args.complete_source,
        complete_target=args.complete_target,
        complete_instance=args.complete_instance,
    )
