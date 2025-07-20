import datasets
import argparse
import os
from devil_in_details.utils import save_jsonl

intent_lable2id = {
    "AddToPlaylist": 0,
    "BookRestaurant": 1,
    "PlayMusic": 2,
    "RateBook": 3,
    "SearchCreativeWork": 4,
    "SearchScreeningEvent": 5,
    "alarm/cancel_alarm": 6,
    "alarm/modify_alarm": 7,
    "alarm/set_alarm": 8,
    "alarm/show_alarms": 9,
    "alarm/snooze_alarm": 10,
    "alarm/time_left_on_alarm": 11,
    "reminder/cancel_reminder": 12,
    "reminder/set_reminder": 13,
    "reminder/show_reminders": 14,
    "weather/find": 15,
}

intent_id2lable = {
    0: "AddToPlaylist",
    1: "BookRestaurant",
    2: "PlayMusic",
    3: "RateBook",
    4: "SearchCreativeWork",
    5: "SearchScreeningEvent",
    6: "alarm/cancel_alarm",
    7: "alarm/modify_alarm",
    8: "alarm/set_alarm",
    9: "alarm/show_alarms",
    10: "alarm/snooze_alarm",
    11: "alarm/time_left_on_alarm",
    12: "reminder/cancel_reminder",
    13: "reminder/set_reminder",
    14: "reminder/show_reminders",
    15: "weather/find",
}

label2id = {
    "O": 0,
    "B-album": 1,
    "I-album": 2,
    "B-artist": 3,
    "I-artist": 4,
    "B-best_rating": 5,
    "I-best_rating": 6,
    "B-condition_description": 7,
    "I-condition_description": 8,
    "B-condition_temperature": 9,
    "I-condition_temperature": 10,
    "B-cuisine": 11,
    "I-cuisine": 12,
    "B-datetime": 13,
    "I-datetime": 14,
    "B-entity_name": 15,
    "I-entity_name": 16,
    "B-facility": 17,
    "I-facility": 18,
    "B-genre": 19,
    "I-genre": 20,
    "B-location": 21,
    "I-location": 22,
    "B-movie_name": 23,
    "I-movie_name": 24,
    "B-movie_type": 25,
    "I-movie_type": 26,
    "B-music_item": 27,
    "I-music_item": 28,
    "B-object_location_type": 29,
    "I-object_location_type": 30,
    "B-object_name": 31,
    "I-object_name": 32,
    "B-object_part_of_series_type": 33,
    "I-object_part_of_series_type": 34,
    "B-object_select": 35,
    "I-object_select": 36,
    "B-object_type": 37,
    "I-object_type": 38,
    "B-party_size_description": 39,
    "I-party_size_description": 40,
    "B-party_size_number": 41,
    "I-party_size_number": 42,
    "B-playlist": 43,
    "I-playlist": 44,
    "B-rating_unit": 45,
    "I-rating_unit": 46,
    "B-rating_value": 47,
    "I-rating_value": 48,
    "B-recurring_datetime": 49,
    "I-recurring_datetime": 50,
    "B-reference": 51,
    "I-reference": 52,
    "B-reminder/todo": 53,
    "I-reminder/todo": 54,
    "B-restaurant_name": 55,
    "I-restaurant_name": 56,
    "B-restaurant_type": 57,
    "I-restaurant_type": 58,
    "B-served_dish": 59,
    "I-served_dish": 60,
    "B-service": 61,
    "I-service": 62,
    "B-sort": 63,
    "I-sort": 64,
    "B-track": 65,
    "I-track": 66,
    "B-weather/attribute": 67,
    "I-weather/attribute": 68,
}

ID2LABEL = {
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
}


def prepare_masakhaner(outdir: str = "data/original/"):
    # Get source data
    outdir = outdir / "masakhaner"
    os.makedirs(outdir, exist_ok=True)

    for split in ["train", "test", "validation"]:
        data = datasets.load_dataset("conll2003", split=split)
        # Replace MISC entity
        ner_tag_column = []
        for i, line in enumerate(data):
            ner_tags = []
            for j, tag in enumerate(line["ner_tags"]):
                if tag == 7 or tag == 8:
                    ner_tags.append(0)
                else:
                    ner_tags.append(tag)
            ner_tag_column.append(ner_tags)
        data = data.remove_columns("ner_tags")
        data = data.add_column("ner_tags", ner_tag_column)
        if split == "validation":
            split = "val"

        outfile = os.path.join(outdir, f"{split}-en.jsonl")
        data.to_json(outfile, force_ascii=False)

    for (
        lg
    ) in "bam ewe fon hau ibo kin lug luo mos nya sna swa tsn twi wol xho yor zul".split():
        for split in ["test", "validation"]:
            data = datasets.load_dataset("masakhane/masakhaner2", lg, split=split)
            # Replace Date entity
            ner_tag_column = []
            for i, line in enumerate(data):
                ner_tags = []
                for j, tag in enumerate(line["ner_tags"]):
                    if tag == 7 or tag == 8:
                        ner_tags.append(0)
                    else:
                        ner_tags.append(tag)
                ner_tag_column.append(ner_tags)
            data = data.remove_columns("ner_tags")
            data = data.add_column("ner_tags", ner_tag_column)
            if split == "validation":
                split = "val"

            outfile = os.path.join(outdir, f"{split}-{lg}.jsonl")
            data.to_json(outfile, force_ascii=False)


def seqs2data(tabular_file: str, skip_first_line: bool = False):
    # Copied from https://github.com/machamp-nlp/machamp/blob/master/machamp/readers/read_sequence.py#L192
    """
    Reads a conll-like file. We do not base the comment identification on
    the starting character being a '#' , as in some of the datasets we used
    the words where in column 0, and could start with a `#'. Instead we start
    at the back, and see how many columns (tabs) the file has. Then we judge
    any sentences at the start which do not have this amount of columns (tabs)
    as comments. Returns both the read column data as well as the full data.

    Parameters
    ----------
    tabular_file: str
        The path to the file to read.
    skip_first_line: bool
        In some csv/tsv files, the heads are included in the first row.
        This option let you skip these.

    Returns
    -------
    full_data: List[List[str]]
        A list with an instance for each token, which is represented as
        a list of strings (split by '\t'). This variable includes the
        comments in the beginning of the instance.
    instance_str: List[List[str]]
        A list with an instance for each token, which is represented as
        a list of strings (split by '\t'). This variable does not include
        the comments in the beginning of the instance.
    """
    sent = []
    for line in open(tabular_file, mode="r", encoding="utf-8"):
        if skip_first_line:
            skip_first_line = False
            continue
        # because people use paste command, which includes empty tabs
        if len(line) < 2 or line.replace("\t", "") in ["" "\n"]:
            if len(sent) == 0:
                continue
            num_cols = len(sent[-1])
            beg_idx = 0
            for i in range(len(sent)):
                back_idx = len(sent) - 1 - i
                if len(sent[back_idx]) == num_cols:
                    beg_idx = len(sent) - 1 - i
            yield sent[beg_idx:], sent
            sent = []
        else:
            if line.startswith("# text"):  # because tab in UD_Munduruku-TuDeT
                line = line.replace("\t", " ")
            sent.append([token for token in line.rstrip("\n").split("\t")])

    # adds the last sentence when there is no empty line
    if len(sent) != 0 and sent != [""]:
        num_cols = len(sent[-1])
        beg_idx = 0
        for i in range(len(sent)):
            back_idx = len(sent) - 1 - i
            if len(sent[back_idx]) == num_cols:
                beg_idx = len(sent) - 1 - i
        yield sent[beg_idx:], sent


def prepare_xsid(outdir: str = "data/original/"):

    input_dir = f"{outdir}/raw/xSID-0.5"
    outdir = f"{outdir}/xsid"
    os.makedirs(outdir, exist_ok=True)

    all_tags = set()
    for lang in "ar da de-st de en id it kk nl sr tr zh".split():
        for split in "valid test".split():  # "train valid test"
            output_data = []
            broken_instance = set()
            if split == "train" and lang != "en":
                continue
            data = seqs2data(f"{input_dir}/{lang}.{split}.conll")
            for i, row in enumerate(data):
                output_row = {"tokens": [], "entity_tags": []}
                tokens_and_labels = row[0]
                intent_raw = tokens_and_labels[0][2]
                # Skip these intents ==> Those are not in the evaluation data
                if intent_raw in ["weather/checkSunset", "weather/checkSunrise"]:
                    continue
                intent_label = intent_lable2id.get(intent_raw, None)
                if intent_label == None:
                    raise AssertionError
                output_row["intent"] = intent_label
                for j, token_and_label in enumerate(tokens_and_labels):
                    if token_and_label[1] != "":
                        output_row["tokens"].append(token_and_label[1])
                    else:
                        continue
                    entity_tag_raw = token_and_label[3]
                    if entity_tag_raw[1:] == "-reference-part":
                        entity_tag_raw = f"{entity_tag_raw[0]}-reference"
                    # Map unused labels to O
                    if entity_tag_raw[2:] in [
                        "alarm/alarm_modifier",
                        "negation",
                        "timer/attributes",
                        "weather/temperatureUnit",
                        "news/type",
                        "reminder/reminder_modifier",
                    ]:
                        entity_tag_raw = "O"
                    # Weird error in the data (check the previous instance)
                    if entity_tag_raw == "Orecurring_datetime":
                        broken_instance.add(i)
                        break
                    entity_tag = label2id.get(entity_tag_raw, None)
                    all_tags.add(entity_tag)
                    if entity_tag == None:
                        raise AssertionError
                    output_row["entity_tags"].append(entity_tag)
                if i in broken_instance:
                    continue
                assert len(output_row["entity_tags"]) == len(output_row["tokens"])
                output_data.append(output_row)

            # Deduplicate the training data
            hashed_example = set()
            del_indices = []
            if split == "train":
                for i, row in enumerate(output_data):
                    # Concatenate tokens and intent to find duplicates
                    tokens = " ".join(row["tokens"] + [str(row["intent"])])
                    if tokens in hashed_example:
                        del_indices.append(i)
                    else:
                        hashed_example.add(tokens)
                for i in sorted(del_indices, reverse=True):
                    del output_data[i]

            # Posprocess the filename
            if split == "valid":
                split = "val"

            save_jsonl(data=output_data, filepath=f"{outdir}/{split}-{lang}.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="data/original/", help="Output directory")

    args = parser.parse_args()

    prepare_masakhaner(args.outdir)
    prepare_xsid(args.outdir)
