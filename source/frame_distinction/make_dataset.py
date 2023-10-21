import argparse
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from vsc.data_utils import read_jsonl, write_jsonl


def filter_examples(df, min_lu, max_lu, min_text, max_text):
    output_list = []
    for verb in tqdm(sorted(set(df["verb"]))):
        df_verb = df[df["verb"] == verb]
        df_lu_count = df_verb.groupby("lu_id").count()
        df_lu_count = df_lu_count.sort_values("text_widx", ascending=False)
        df_lu_count = df_lu_count[:max_lu]
        df_lu_count = df_lu_count[df_lu_count["text_widx"] >= min_text]
        if len(df_lu_count) < min_lu:
            continue
        for lu_id in list(df_lu_count.index):
            df_lu = df_verb[df_verb["lu_id"] == lu_id][:max_text]
            output_list += df_lu.to_dict("records")
    return pd.DataFrame(output_list)


def decide_sets(df):
    verb_list = sorted(set(df["verb"]))
    random.shuffle(verb_list)

    dev_list, test_list = verb_list[:-120], verb_list[-120:]
    verb2sets = {v: "dev" for v in dev_list}
    verb2sets.update({v: "test" for v in test_list})
    return df["verb"].map(verb2sets)


def main(args):
    random.seed(args.random_state)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(read_jsonl(args.input_file))
    df = df.sample(n=len(df), random_state=0)

    df_output = filter_examples(
        df, args.min_lu, args.max_lu, args.min_text, args.max_text
    )
    df_output["sets"] = decide_sets(df_output)

    write_jsonl(
        df_output.to_dict("records"), args.output_dir / "exemplars.jsonl"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--min_lu", type=int, default=2)
    parser.add_argument("--max_lu", type=int, default=10)
    parser.add_argument("--min_text", type=int, default=20)
    parser.add_argument("--max_text", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=0)
    args = parser.parse_args()
    print(args)
    main(args)
