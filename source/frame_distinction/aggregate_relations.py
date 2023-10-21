import argparse
from pathlib import Path

import pandas as pd

from vsc.data_utils import read_jsonl, write_json


def make_v2f(df):
    v2f = {}
    for df_dict in df.groupby("verb").agg(set).reset_index().to_dict("records"):
        v2f[df_dict["verb"]] = list(df_dict["frame_name"])
    return v2f


def make_f2f(df):
    f2f = {}
    for df_dict in (
        df.groupby("target_frame").agg(set).reset_index().to_dict("records")
    ):
        f2f[df_dict["target_frame"]] = df_dict["related_frame"]
    return f2f


def split_verb_rel(v2f, f2f):
    wo_rel_list, w_rel_list = [], []
    for v, f in v2f.items():
        if len(f) != 2:
            continue
        rel = 0
        if f[0] in f2f:
            if f[1] in f2f[f[0]]:
                rel = 1
        if f[1] in f2f:
            if f[0] in f2f[f[1]]:
                rel = 1

        if rel == 0:
            wo_rel_list.append(v)
        else:
            w_rel_list.append(v)
    return wo_rel_list, w_rel_list


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_dataset = pd.DataFrame(read_jsonl(args.input_file))
    df_f2f = pd.DataFrame(read_jsonl(args.input_f2f_file))
    df_score = pd.DataFrame(read_jsonl(args.input_score_file))

    df_dataset = df_dataset[df_dataset["sets"] == args.sets]

    v2f = make_v2f(df_dataset)
    f2f = make_f2f(df_f2f)
    wo_rel_list, w_rel_list = split_verb_rel(v2f, f2f)

    mean_wo = df_score[df_score["verb"].isin(wo_rel_list)]["score"].mean()
    mean_w = df_score[df_score["verb"].isin(w_rel_list)]["score"].mean()
    score_dict = {
        "gr_wo_rel": mean_wo,
        "gr_w_rel": mean_w,
        "diff": mean_wo - mean_w,
        "n_wo_verbs": len(wo_rel_list),
        "n_w_verbs": len(w_rel_list),
    }
    print(score_dict)

    file = f"score_{args.layer}.json"
    write_json(score_dict, args.output_dir / file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=Path)
    parser.add_argument("--input_score_file", type=Path)
    parser.add_argument("--input_f2f_file", type=Path)
    parser.add_argument("--output_dir", type=Path)

    parser.add_argument(
        "--model_name",
        type=str,
        choices=[
            "all-in-one-cluster",
            "elmo",
            "bert-base-uncased",
            "bert-large-uncased",
            "roberta-base",
            "albert-base-v2",
            "gpt2",
            "xlnet-base-cased",
        ],
    )
    parser.add_argument("--layer", type=str, default="best")
    parser.add_argument("--sets", type=str, choices=["dev", "test"])
    args = parser.parse_args()
    print(args)
    main(args)
