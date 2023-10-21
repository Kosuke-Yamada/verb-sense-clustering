# -*-coding:utf-8-*-

import argparse
import json
import os

import pandas as pd


def parse_args():
    MODEL_NAME = [
        "all-in-one-cluster",
        "elmo",
        "bert-base-uncased",
        "bert-large-uncased",
        "roberta-base",
        "albert-base-v2",
        "gpt2",
        "xlnet-base-cased",
    ][0]
    LAYER = -1
    SETS = ["dev", "test"][1]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="../data/experiment_frame_distinction/dataset/framenet",
    )
    parser.add_argument(
        "--score_path",
        type=str,
        default="../data/experiment_frame_distinction/frame_distinction/framenet",
    )
    parser.add_argument(
        "--f2f_path",
        type=str,
        default="../data/preprocessing/relations",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="../data/experiment_frame_distinction/relations",
    )

    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--sets", type=str, default=SETS)
    return parser.parse_args()


def make_dir_path(dataset_path, score_path, f2f_path, output_path, model_name, sets):
    path_dict = {
        "input_dataset": dataset_path,
        "input_score": "/".join([score_path, model_name, sets]),
        "input_f2f": f2f_path,
        "output": "/".join([output_path, model_name, sets]),
    }
    for key, path in path_dict.items():
        path_dict[key] = os.path.abspath(path) + "/"
        if "output" in key:
            if not os.path.isdir(path_dict[key]):
                os.makedirs(path_dict[key])
    return path_dict


def make_v2f(df):
    v2f = {}
    for df_dict in df.groupby("verb").agg(set).reset_index().to_dict("records"):
        v2f[df_dict["verb"]] = list(df_dict["frame_name"])
    return v2f


def make_f2f(df):
    f2f = {}
    for df_dict in df.groupby("target_frame").agg(set).reset_index().to_dict("records"):
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


def main():
    args = parse_args()
    path_dict = make_dir_path(
        args.dataset_path,
        args.score_path,
        args.f2f_path,
        args.output_path,
        args.model_name,
        args.sets,
    )

    df_dataset = pd.read_json(
        path_dict["input_dataset"] + "exemplars.jsonl", orient="records", lines=True
    )
    df_f2f = pd.read_json(
        path_dict["input_f2f"] + "f2f_list.jsonl", orient="records", lines=True
    )
    df_score = pd.read_json(
        path_dict["input_score"] + "verb_scores_" + str(args.layer).zfill(2) + ".jsonl",
        orient="records",
        lines=True,
    )

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

    file_path = "score_" + str(args.layer).zfill(2) + ".json"
    with open(path_dict["output"] + file_path, "w") as f:
        json.dump(score_dict, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
