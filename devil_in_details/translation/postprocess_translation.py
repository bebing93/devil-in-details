import re
import os
import argparse
import logging
from devil_in_details.utils import load_jsonl, save_jsonl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_translation(text):
    text = (
        text.replace("\u200c", " ")
        .replace("\u200b", " ")
        .replace("\u200d", " ")
        .replace("\u200e", " ")
        .replace("\xa0", " ")
    )
    text = re.sub(r"\s+", " ", text)
    return text


def postprocess_bio(
    input_path: str,
    lang: str,
):
    in_dir_name = os.path.dirname(input_path)
    in_file_name = os.path.splitext(os.path.basename(input_path))[0]
    output_file_name = f"{in_file_name}-processed"

    # Read input data
    input_data = load_jsonl(input_path)

    out_data = []
    for line in input_data:
        line = clean_translation(line["translation"][lang])
        out_data.append({"translation": {lang: line}})

    # Write output
    save_jsonl(out_data, f"{in_dir_name}/{output_file_name}.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to the translated data file")
    parser.add_argument("lang", help="Language that was translated to")

    args = parser.parse_args()

    postprocess_bio(args.input_path, args.lang)
