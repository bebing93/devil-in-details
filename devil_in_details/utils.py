import json
import argparse
from collections import defaultdict
from typing import List, Dict, Any, Tuple
import logging
import os
import torch
import random
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading {filepath}: {e}")
        raise


def load_text_lines(filepath: str) -> List[str]:
    """Load text file lines."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return [line.strip() for line in f]
    except FileNotFoundError as e:
        logger.error(f"Error loading {filepath}: {e}")
        raise


def save_text_lines(data: List[str], filepath: str) -> None:
    """Save list of strings to a text file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        for line in data:
            f.write(f"{line}\n")


def save_jsonl(data: List[Dict[str, Any]], filepath: str) -> None:
    """Save data to JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


def parse_alignment_line(line: str) -> List[List[int]]:
    """Parse alignment line into list of [source_idx, target_idx] pairs."""
    if not line.strip():
        return []

    try:
        alignments = []
        for alignment in line.split():
            src_idx, trg_idx = map(int, alignment.split("-"))
            alignments.append([src_idx, trg_idx])
        return alignments
    except (ValueError, IndexError) as e:
        logger.error(f"Error parsing alignment line: {line}")
        raise e


def build_alignment_mapping(
    alignment_line: List[List[int]], inverse=False
) -> Dict[int, List[int]]:
    """Build source-to-target alignment mapping.

    Args:
        alignment_line: List of [source_idx, target_idx] pairs
        src2trg: If true, maps from source idx to target idx else vice-versa

    Returns:
        Dictionary mapping source indices to sorted target indices:     <source_index>: [<target_index>, <target_index>, ...]
    """

    src2trg = defaultdict(list)
    for src_idx, trg_idx in alignment_line:
        if inverse:
            src2trg[trg_idx].append(src_idx)
        else:
            src2trg[src_idx].append(trg_idx)

    return {k: sorted(v) for k, v in src2trg.items()}


def extract_entity_indices(labels: List[int]) -> Tuple[List[List[int]], List[int]]:
    """Extract entity indices and types from BIO labels.

    Args:
        labels: List of BIO tag integers

    Returns:
        Tuple of (entity_indices, entity_types)
    """
    entity = []
    entities = []
    entity_types = []
    in_entity = False
    for j, tag in enumerate(labels):
        if tag == 0:  # O Tag
            if in_entity == True:
                # End of previous entity reached
                entities.append(entity)
                entity_types.append(labels[entity[0]])
                in_entity = False
            entity = []
            continue
        elif tag % 2 == 1:  # B-Tag
            if in_entity == True:
                # End of previous entity reached
                entities.append(entity)
                entity_types.append(labels[entity[0]])

            # Start new entity
            in_entity = True
            entity = [j]
        elif tag % 2 == 0:  # I-Tag
            if in_entity == True:
                # Within current entity
                entity.append(j)
            else:
                # Orphaned I tag - treat as beginning of new entity
                in_entity = True
                entity = [j]

    # Handle end of sequence entity
    if entity:
        entities.append(entity)
        entity_types.append(labels[entity[0]])
    return entities, entity_types


def load_logits_with_retry(
    logit_path: str, max_attempts: int = 30, max_sleep_time: int = 30
) -> torch.Tensor:
    """
    Load logits from file with retry mechanism.

    Args:
        logit_path: Path to the logits file
        max_attempts: Maximum number of retry attempts

    Returns:
        Loaded logits tensor

    Raises:
        FileNotFoundError: If file cannot be loaded after max attempts
    """
    for attempt in range(max_attempts):
        try:
            logger.info(
                f"Loading logits from {logit_path} (attempt {attempt + 1}/{max_attempts})"
            )
            logits = torch.load(logit_path, map_location=torch.device("cpu"))
            logger.info(f"Successfully loaded logits from {logit_path}")
            return logits
        except Exception as e:
            logger.warning(f"Failed to load {logit_path} on attempt {attempt + 1}: {e}")
            if attempt < max_attempts - 1:
                sleep_time = random.randint(1, max_sleep_time)
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                raise FileNotFoundError(
                    f"Could not load logits from {logit_path} after {max_attempts} attempts"
                )


def str_to_bool(v: str) -> bool:
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
