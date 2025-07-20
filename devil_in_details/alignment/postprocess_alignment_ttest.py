import argparse
import logging
from devil_in_details.utils import load_jsonl, load_text_lines, save_jsonl, parse_alignment_line

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def postprocess_bio_alignment(
    source_file: str,
    target_file: str,
    alignment_file: str,
    out_file: str,
    text_column: str = "tokens",
    tag_name: str = "ner_tags",
):

    logger.info("Loading data files...")
    # Read data
    source_data = load_text_lines(source_file)  # i.e., translated English data
    target_data = load_jsonl(target_file)
    alignment_out_data = load_text_lines(alignment_file)

    # Validate data consistency
    if not (len(source_data) == len(target_data) == len(alignment_out_data)):
        raise ValueError(
            f"Data length mismatch: source={len(source_data)}, "
            f"target={len(target_data)}, alignment={len(alignment_out_data)}"
        )

    processed_data = []
    for idx, (src_line, trg_line, alignment_line) in enumerate(
        zip(source_data, target_data, alignment_out_data)
    ):
        try:
            # Parse inputs
            trg_tokens = trg_line[text_column]
            src_tokens = src_line.split()
            alignment_pairs = parse_alignment_line(alignment_line)

            # Create output item
            output_item = trg_line.copy()
            output_item["alignment"] = alignment_pairs
            output_item[f"org_{text_column}"] = trg_tokens
            output_item[text_column] = src_tokens
            output_item[f"org_{tag_name}"] = trg_line[tag_name]
            # Create dummy source labels (most fine-tuning scripts expect labels)
            output_item[tag_name] = [0] * len(src_tokens)

            processed_data.append(output_item)

        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            raise AssertionError

    # Report results
    total_items = len(processed_data)

    logger.info(f"Processing complete:")
    logger.info(f"  Total items: {total_items}")

    # Save results
    logger.info(f"Saving results to {out_file}")
    save_jsonl(processed_data, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Align BIO tags between source (usually translated English data) and target (usually clean target language data) using word alignments"
    )
    parser.add_argument(
        "source_file", help="Source data contain the English translations"
    )
    parser.add_argument(
        "target_file", help="Target data file containg the clean target language data"
    )
    parser.add_argument(
        "alignment_file",
        help="Word alignment file mapping from source to target (i.e., English to target language)",
    )
    parser.add_argument("out_file", help="Output file path (JSONL format)")
    parser.add_argument(
        "--text_column", default="tokens", help="Name of tokens column in source file"
    )
    parser.add_argument(
        "--tag_name", default="ner_tags", help="Name of label column in source file"
    )

    args = parser.parse_args()

    postprocess_bio_alignment(
        source_file=args.source_file,
        target_file=args.target_file,
        alignment_file=args.alignment_file,
        out_file=args.out_file,
        text_column=args.text_column,
        tag_name=args.tag_name,
    )
