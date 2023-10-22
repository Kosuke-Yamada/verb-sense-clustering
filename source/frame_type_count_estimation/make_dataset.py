import argparse
import os
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from vsc.data_utils import read_jsonl, write_jsonl


def filter_examples(df, min_lu, max_lu, min_text, max_text):
    output_list1, output_list2, lu_count1_list = [], [], []
    for verb in tqdm(sorted(set(df["verb"]))):
        df_verb = df[df["verb"] == verb]
        df_lu_count = df_verb.groupby("lu_id").count()
        df_lu_count = df_lu_count.sort_values("text_widx", ascending=False)[
            :max_lu
        ]
        df_lu_count = df_lu_count[df_lu_count["text_widx"] >= min_text]
        if len(df_lu_count) >= min_lu:
            for lu_id in list(df_lu_count.index):
                df_lu = df_verb[df_verb["lu_id"] == lu_id][:max_text]
                output_list2 += df_lu.to_dict("records")
        elif len(df_lu_count) == 1:
            output_list1 += df_verb[:max_text].to_dict("records")
            lu_count1_list.append(
                {"verb": verb, "count": df_lu_count["target_widx"].values[0]}
            )

    temp_df_output1 = pd.DataFrame(output_list1)
    df_output2 = pd.DataFrame(output_list2)
    df_lu_count1 = pd.DataFrame(lu_count1_list).sort_values(
        "count", ascending=False
    )[: len(set(df_output2["verb"]))]
    df_output1 = temp_df_output1[
        temp_df_output1["verb"].isin(df_lu_count1["verb"])
    ]
    return df_output1, df_output2


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

    df_output1, df_output2 = filter_examples(
        df, args.min_lu, args.max_lu, args.min_text, args.max_text
    )
    df_output1["sets"] = decide_sets(df_output1)
    df_output2["sets"] = decide_sets(df_output2)

    df_output = pd.concat([df_output1, df_output2], axis=0)
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
