import argparse
import re
import os
import logging
from devil_in_details.utils import load_jsonl, save_jsonl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def judge_if_Chinese_and_Japanese_token(pretokenized_text):
    """
    Adpoted from https://github.com/edchengg/easyproject/blob/main/ner/decode_marker_conll.py

    Judge each token in a pretokenized sent if Chinese or Japanese
    Input:
        a sentence. ['增加', '一项', '提醒', '在', '今天', '下午', '4']
    Output:
        a 0, 1 list, 1 means is Chinese token, 0 means not.
        [1, 1, 1, 1, 1, 1, 0]
    """
    "U+0E00..U+0E7F"
    return [
        (
            1
            if re.search(
                r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f\u0E00-\u0E7F]+",
                i,
            )
            != None
            else 0
        )
        for i in pretokenized_text
    ]


def preprocess(
    input_path: str,
    output_path: str,
    column: str,
    lang: str,
):

    in_dir_name = os.path.dirname(input_path)
    in_file_name = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.dirname(output_path)

    # Prepare output dir
    os.makedirs(out_dir, exist_ok=True)

    # Read data
    input_data = load_jsonl(f"{in_dir_name}/{in_file_name}.jsonl")

    logger.info(f"*** Preprocessing for file: {input_path}, column: {column}")

    out_data = []
    for line in input_data:
        tokens = line[column]

        if lang == "zh":
            # Merge Chinese tokens without whitespace
            Chinese_or_Eng_list = judge_if_Chinese_and_Japanese_token(tokens)
            text = tokens[0]
            if len(tokens) > 1:
                for idx, token in enumerate(tokens):
                    if idx == 0:
                        # Already part of text
                        continue
                    # 1. Case: Previous token is not Chinese or current token is not Chinese
                    if (
                        Chinese_or_Eng_list[idx - 1] == 0
                        or Chinese_or_Eng_list[idx] == 0
                    ):
                        text = text + " " + token
                    # 2. Case: Previous token is Chinese and current token is Chinese
                    elif (
                        Chinese_or_Eng_list[idx] == 1
                        and Chinese_or_Eng_list[idx - 1] == 1
                    ):
                        text = text + token
                    else:
                        raise AssertionError

        else:
            text = " ".join(tokens)

        out_data.append({"translation": {lang: text}})

    save_jsonl(out_data, output_path)

    logger.info(f"*** Preprocessed file written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path",
        help="Path to the original input data (clean data used for translation)",
    )
    parser.add_argument(
        "output_path",
        help="Path to the preprocessed original input data (clean data used for translation)",
    )
    parser.add_argument(
        "column",
        help="Text column in the dataset that contains the text for translation",
    )
    parser.add_argument("lang", help="Language of the data (special handling for zh)")
    args = parser.parse_args()

    preprocess(args.input_path, args.output_path, args.column, args.lang)
