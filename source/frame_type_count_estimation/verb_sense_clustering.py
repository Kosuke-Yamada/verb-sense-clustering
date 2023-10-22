import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from vsc.clustering import VerbSenseClustering
from vsc.data_utils import read_json, read_jsonl, write_json, write_jsonl
from vsc.scoring import eval_confusion_matrix


def make_df_abic(vsc, df, vec_array, max_n_frames):
    print("make_df_abic")
    abic_list = []
    for verb in tqdm(sorted(set(df["verb"]))):
        df_verb = df[df["verb"] == verb].copy()
        vec_verb = vec_array[df_verb.index]

        df_abic = vsc.run_clustering_abic(vec_verb, max_n_frames)
        df_abic["verb"] = verb
        df_abic["n_frames"] = len(set(df_verb["frame_name"]))
        abic_list += df_abic.to_dict("records")
    return pd.DataFrame(abic_list)


def decide_params(df_abic, max_n_frames):
    print("abc")
    min_diff_n, min_c = 1e10, -1
    for c in tqdm([i / 10 for i in range(51)]):
        _, df_output = aggregate_abic(df_abic, c, max_n_frames, max_n_frames)
        diff_n = abs(sum(df_output["n_frames"]) - sum(df_output["n_clusters"]))
        if min_diff_n >= diff_n:
            min_diff_n = diff_n
            min_c = c
    return {"diff_n": diff_n, "min_c": min_c}


def aggregate_abic(df, c, max_n_frames, max_n_clusters):
    df["bic"] = df["first"] + df["second"] * c
    df_min = df.loc[df.groupby("verb")["bic"].idxmin()]
    df_cm = df_min.pivot_table(
        index="n_frames", columns="n_clusters", aggfunc="size", fill_value=0
    )
    return df_cm, df_min[["verb", "n_frames", "n_clusters"]]


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.layer == "best":
        best_score, layer = -1, -1
        for file in args.input_dev_dir.glob("score-*.json"):
            score = read_json(file)["score"]
            if best_score < score:
                best_score = score
                layer = file.name.split("-")[1].split(".json")[0]
    else:
        layer = args.layer

    df = pd.DataFrame(read_jsonl(args.input_dir / "exemplars.jsonl"))
    vec_array = np.load(args.input_dir / f"vec-{layer}.npz", allow_pickle=True)[
        "vec"
    ]

    max_n_frames_dev = (
        df[df["sets"] == "dev"]
        .groupby("verb")["frame_name"]
        .agg(set)
        .apply(len)
        .max()
    )

    df = df[df["sets"] == args.sets]
    vec_array = vec_array[df.index]
    df = df.reset_index()

    vsc = VerbSenseClustering(args.n_seeds, args.covariance_type)
    df_abic = make_df_abic(vsc, df, vec_array, max_n_frames_dev)

    params = {"min_c": 1}
    if args.sets == "dev":
        max_n_frames = max_n_frames_dev
        max_n_clusters = max_n_frames_dev
        if args.criterion == "abic":
            params = decide_params(df_abic, max_n_frames_dev)
            file = f"params-{args.layer}.json"
            write_json(params, args.output_dir / file)

    elif args.sets == "test":
        max_n_frames = (
            df.groupby("verb")["frame_name"].agg(set).apply(len).max()
        )
        max_n_clusters = max_n_frames_dev
        if args.criterion == "abic":
            params = read_json(args.input_params_file)

    df_cm, df_output = aggregate_abic(
        df_abic, params["min_c"], max_n_frames, max_n_clusters
    )
    score_dict = eval_confusion_matrix(df_output)

    file = f"confusion_matrix-{args.layer}.jsonl"
    write_jsonl(df_cm.to_dict("records"), args.output_dir / file)

    file = f"n_frames_clusters-{args.layer}.jsonl"
    write_jsonl(df_output.to_dict("records"), args.output_dir / file)

    file = f"score-{args.layer}.json"
    write_json(score_dict, args.output_dir / file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path)
    parser.add_argument("--input_dev_dir", type=Path)
    parser.add_argument("--input_params_file", type=Path)
    parser.add_argument("--output_dir", type=Path)

    parser.add_argument(
        "--model_name",
        type=str,
        default=[
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
    parser.add_argument("--layer", type=str, default="00")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--covariance_type", type=str, default="spherical")
    parser.add_argument("--sets", type=str, default=["dev", "test"])
    parser.add_argument("--criterion", type=str, choices=["bic", "abic"])
    args = parser.parse_args()
    main(args)
