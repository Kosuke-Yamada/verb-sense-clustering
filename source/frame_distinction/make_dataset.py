# -*-coding:utf-8-*-

import argparse
import os
import random

import pandas as pd
from tqdm import tqdm


def parse_args():
    RESOURCE = ["framenet", "propbank"][1]

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="../data/preprocessing")
    parser.add_argument(
        "--output_path",
        type=str,
        default="../data/experiment_frame_distinction/dataset",
    )
    parser.add_argument("--resource", type=str, default=RESOURCE)
    parser.add_argument("--min_lu", type=int, default=2)
    parser.add_argument("--max_lu", type=int, default=10)
    parser.add_argument("--min_text", type=int, default=20)
    parser.add_argument("--max_text", type=int, default=100)
    return parser.parse_args()


def make_dir_path(input_path, output_path, resource):
    path_dict = {
        "input": "/".join([input_path, resource]),
        "output": "/".join([output_path, resource]),
    }
    for key, path in path_dict.items():
        path_dict[key] = os.path.abspath(path) + "/"
        if "output" in key:
            if not os.path.isdir(path_dict[key]):
                os.makedirs(path_dict[key])
    return path_dict


def filter_examples(df, min_lu, max_lu, min_text, max_text):
    output_list = []
    for verb in tqdm(sorted(set(df["verb"]))):
        df_verb = df[df["verb"] == verb]
        df_lu_count = df_verb.groupby("lu_id").count()
        df_lu_count = df_lu_count.sort_values("text_widx", ascending=False)[:max_lu]
        df_lu_count = df_lu_count[df_lu_count["text_widx"] >= min_text]
        if len(df_lu_count) < min_lu:
            continue
        for lu_id in list(df_lu_count.index):
            df_lu = df_verb[df_verb["lu_id"] == lu_id][:max_text]
            output_list += df_lu.to_dict("records")
    return pd.DataFrame(output_list)


def decide_sets(df):
    verb_list = sorted(set(df["verb"]))
    random.seed(0)
    random.shuffle(verb_list)

    dev_list, test_list = verb_list[:-120], verb_list[-120:]
    verb2sets = {v: "dev" for v in dev_list}
    verb2sets.update({v: "test" for v in test_list})
    return df["verb"].map(verb2sets)


def main():
    args = parse_args()
    path_dict = make_dir_path(args.input_path, args.output_path, args.resource)

    df = pd.read_json(
        path_dict["input"] + "exemplars.jsonl", orient="records", lines=True
    )
    df = df.sample(n=len(df), random_state=0)

    df_output = filter_examples(
        df, args.min_lu, args.max_lu, args.min_text, args.max_text
    )
    df_output["sets"] = decide_sets(df_output)

    df_output.to_json(
        path_dict["output"] + "exemplars.jsonl",
        orient="records",
        force_ascii=False,
        lines=True,
    )


if __name__ == "__main__":
    main()
