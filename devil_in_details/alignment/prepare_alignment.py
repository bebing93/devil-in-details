import argparse
import jieba
from sacremoses import MosesTokenizer
from devil_in_details.utils import load_jsonl, save_text_lines


def prepare_alignment_bio(
    original_file,  # path to jsonl
    original_text_column,
    translated_lang,
    translated_file,  # path to jsonl
    original_out_file,
    translated_out_file,
    tokenizer="whitespace",
):

    # Parse input
    original_data = load_jsonl(original_file)
    # Extract original text from the specified column
    original_data = [line[original_text_column] for line in original_data]

    translated_data = load_jsonl(translated_file)
    # Extract translated text from the specified column
    translated_data = [line["translation"][translated_lang] for line in translated_data]

    # Validate that both files have the same number of lines
    if len(original_data) != len(translated_data):
        raise ValueError(
            f"original and translated files have different number of lines: {len(original_data)} vs {len(translated_data)}"
        )

    if tokenizer == "moses":
        translated_tokenizer = MosesTokenizer(lang=translated_lang)

    original_alignment_in_lines = []
    translated_alignment_in_lines = []

    for org_line, trans_line in zip(original_data, translated_data):
        org_alignment_in = " ".join(org_line)

        if translated_lang == "zh":
            trans_alignment_in = " ".join(jieba.cut(trans_line))
        elif tokenizer == "moses":
            trans_alignment_in = translated_tokenizer.tokenize(
                trans_line, escape=False, return_str=True
            )
        else:  # whitespace tokenization (default case)
            trans_alignment_in = trans_line

        original_alignment_in_lines.append(org_alignment_in)
        translated_alignment_in_lines.append(trans_alignment_in)

    save_text_lines(original_alignment_in_lines, original_out_file)
    save_text_lines(translated_alignment_in_lines, translated_out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "original_file", help="Path to the original file in jsonl format"
    )
    parser.add_argument(
        "original_text_column", help="Column with the tokens in the original file"
    )
    parser.add_argument(
        "translated_lang",
        help="Translated lang for the Moses tokenizer (we support zh with jieba)",
    )
    parser.add_argument(
        "translated_file", help="Path to the translated file in jsonl format"
    )
    parser.add_argument("org_out_file", help="Path to the original output file")
    parser.add_argument("trans_out_file", help="Path to the translated output file")
    parser.add_argument(
        "--tokenizer",
        default="whitespace",
        help="Set to 'moses' to pretokenize with MosesTokenizer, zh will always be tokenized with jieba)",
    )
    args = parser.parse_args()
    prepare_alignment_bio(
        original_file=args.original_file,
        original_text_column=args.original_text_column,
        translated_lang=args.translated_lang,
        translated_file=args.translated_file,
        original_out_file=args.org_out_file,
        translated_out_file=args.trans_out_file,
        tokenizer=args.tokenizer,
    )
