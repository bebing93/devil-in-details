import argparse
import sys
from typing import Optional, List, Dict
from devil_in_details.utils import save_text_lines, load_logits_with_retry, str_to_bool
import logging

import torch
import datasets
import evaluate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
ID2LABEL = {
    "masakhaner": {
        0: "O",
        1: "B-PER",
        2: "I-PER",
        3: "B-ORG",
        4: "I-ORG",
        5: "B-LOC",
        6: "I-LOC",
    },
    "xsid": {
        0: "O",
        1: "B-album",
        2: "I-album",
        3: "B-artist",
        4: "I-artist",
        5: "B-best_rating",
        6: "I-best_rating",
        7: "B-condition_description",
        8: "I-condition_description",
        9: "B-condition_temperature",
        10: "I-condition_temperature",
        11: "B-cuisine",
        12: "I-cuisine",
        13: "B-datetime",
        14: "I-datetime",
        15: "B-entity_name",
        16: "I-entity_name",
        17: "B-facility",
        18: "I-facility",
        19: "B-genre",
        20: "I-genre",
        21: "B-location",
        22: "I-location",
        23: "B-movie_name",
        24: "I-movie_name",
        25: "B-movie_type",
        26: "I-movie_type",
        27: "B-music_item",
        28: "I-music_item",
        29: "B-object_location_type",
        30: "I-object_location_type",
        31: "B-object_name",
        32: "I-object_name",
        33: "B-object_part_of_series_type",
        34: "I-object_part_of_series_type",
        35: "B-object_select",
        36: "I-object_select",
        37: "B-object_type",
        38: "I-object_type",
        39: "B-party_size_description",
        40: "I-party_size_description",
        41: "B-party_size_number",
        42: "I-party_size_number",
        43: "B-playlist",
        44: "I-playlist",
        45: "B-rating_unit",
        46: "I-rating_unit",
        47: "B-rating_value",
        48: "I-rating_value",
        49: "B-recurring_datetime",
        50: "I-recurring_datetime",
        51: "B-reference",
        52: "I-reference",
        53: "B-reminder/todo",
        54: "I-reminder/todo",
        55: "B-restaurant_name",
        56: "I-restaurant_name",
        57: "B-restaurant_type",
        58: "I-restaurant_type",
        59: "B-served_dish",
        60: "I-served_dish",
        61: "B-service",
        62: "I-service",
        63: "B-sort",
        64: "I-sort",
        65: "B-track",
        66: "I-track",
        67: "B-weather/attribute",
        68: "I-weather/attribute",
    },
}


def ensemble_predictions(
    all_logits: Dict[int, torch.Tensor], replace_second_logits: bool = True
) -> List[torch.Tensor]:
    """
    Create ensemble predictions from multiple logit sets.

    Args:
        all_logits: Dictionary mapping model indices to their logits
        replace_second_logits: Whether to replace unmapped logits in second model

    Returns:
        List of prediction tensors for each sequence
    """
    ensemble_preds = []
    num_sequences = len(all_logits[0])

    for i in range(num_sequences):
        probs = []

        for model_idx, logits in all_logits.items():
            sequence_logits = logits[i]

            # Handle replacement of unmapped logits for second model
            if replace_second_logits and model_idx == 1:
                sequence_logits = [
                    (
                        all_logits[0][i][k]
                        if token_logits[0] == sys.maxsize
                        else token_logits
                    )
                    for k, token_logits in enumerate(sequence_logits)
                ]

            # Convert to probabilities
            sequence_probs = torch.nn.functional.softmax(
                torch.stack(sequence_logits), dim=-1
            )
            probs.append(sequence_probs)

        # Average probabilities across models
        ensemble_probs = torch.stack(probs, dim=0).mean(dim=0)
        predictions = ensemble_probs.argmax(dim=-1)
        ensemble_preds.append(predictions)

    return ensemble_preds


def evaluate_bio(
    task: str,
    dataset_path: str,
    out_score_path: str,
    first_logit_path: str,
    second_logit_path: Optional[str] = None,
    replace_second_logits: bool = True,
    label_column: str = "org_ner_tags",
) -> None:
    """
    Evaluate BIO tagging performance using (ensemble) of models.

    Args:
        task: Task name
        dataset_path: Path to test dataset
        out_score_path: Path to save evaluation scores
        first_logit_path: Path to first model's logits
        second_logit_path: Path to second model's logits (optional)
        replace_second_logits: Whether to replace unmapped logits in second model; if true gives the first models predicitions on these tokens
        label_column: Column name containing labels
    """

    # TODO: Read label lists from file for easier extendability

    # Load test data
    test_data = datasets.load_dataset(
        path=f"json",
        data_files=dataset_path,
        split="train",
    )

    # Load logits
    all_logits = {}

    all_logits[0] = load_logits_with_retry(first_logit_path)

    if second_logit_path:
        all_logits[1] = load_logits_with_retry(second_logit_path)

    # Validate logits compatibility
    if len(all_logits) > 1:
        first_length = len(all_logits[0])
        for idx, logits in all_logits.items():
            if len(logits) != first_length:
                raise ValueError(
                    f"Logit length mismatch: model {idx} has {len(logits)} sequences, expected {first_length}"
                )

    ensemble_preds = ensemble_predictions(all_logits, replace_second_logits)

    pred_labels = [
        [ID2LABEL[task][pred.item()] for pred in pred_sequence]
        for pred_sequence in ensemble_preds
    ]

    # Get the target labels
    true_labels = [
        [ID2LABEL[task][l] for l in labels] for labels in test_data[label_column]
    ]

    if len(pred_labels) != len(true_labels):
        raise ValueError(
            f"Prediction and label length mismatch: {len(pred_labels)} vs {len(true_labels)}"
        )

    metric = evaluate.load("seqeval", keep_in_memory=True)
    score = metric.compute(predictions=pred_labels, references=true_labels)
    f1_score = round(score["overall_f1"] * 100, 2)

    save_text_lines([str(f1_score)], out_score_path)
    logger.info(f"F1 Score: {f1_score}%")
    logger.info(f"Evaluation complete. Results saved to {out_score_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate BIO tagging performance using ensemble of models"
    )
    parser.add_argument("task", help="Task name")
    parser.add_argument("dataset_path", help="Path to test dataset")
    parser.add_argument("out_score_path", help="Path to save evaluation scores")
    parser.add_argument("first_logit_path", help="Path to first model's logits")
    parser.add_argument(
        "--second_logit_path", default=None, help="Path to second model's logits"
    )
    parser.add_argument(
        "--replace_second_logits",
        type=str_to_bool,
        default=True,
        help="Whether to replace unmapped logits in second model",
    )
    parser.add_argument(
        "--label_column", default="org_ner_tags", help="Column name containing labels"
    )

    args = parser.parse_args()

    try:
        evaluate_bio(
            task=args.task,
            dataset_path=args.dataset_path,
            out_score_path=args.out_score_path,
            first_logit_path=args.first_logit_path,
            second_logit_path=args.second_logit_path,
            replace_second_logits=args.replace_second_logits,
            label_column=args.label_column,
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)
