# -*-coding:utf-8-*-

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from vsc.clustering import VerbSenseClustering


def parse_args():
    RESOURCE = ["framenet", "propbank"][0]
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
    LAYER = 0
    SETS = ["dev", "test"][0]
    CRITERION = ["bic", "abic"][0]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="../data/experiment_frame_number_estimation/embeddings",
    )
    parser.add_argument(
        "--dev_path",
        type=str,
        default="../data/experiment_frame_distinction/frame_distinction",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../data/experiment_frame_number_estimation/frame_number_estimation",
    )

    parser.add_argument("--resource", type=str, default=RESOURCE)
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--covariance_type", type=str, default="spherical")
    parser.add_argument("--sets", type=str, default=SETS)
    parser.add_argument("--criterion", type=str, default=CRITERION)
    return parser.parse_args()


def make_dir_path(input_path, dev_path, output_path, resource, model_name, criterion):
    path_dict = {
        "input": "/".join([input_path, resource, model_name]),
        "input_dev": "/".join([dev_path, resource, model_name, "dev"]),
        "output_dev": "/".join([output_path, resource, model_name, criterion, "dev"]),
        "output_test": "/".join([output_path, resource, model_name, criterion, "test"]),
    }
    for key, path in path_dict.items():
        path_dict[key] = os.path.abspath(path) + "/"
        if "output" in key:
            if not os.path.isdir(path_dict[key]):
                os.makedirs(path_dict[key])
    return path_dict


def make_df_abic(vsc, df, vec_array, max_n_frames):
    abic_list = []
    for verb in tqdm(sorted(set(df["verb"]))):
        df_verb = df[df["verb"] == verb].copy()
        vec_verb = vec_array[df_verb.index]

        temp_df_abic = vsc.run_clustering_abic(vec_verb, max_n_frames)
        temp_df_abic["verb"] = verb
        temp_df_abic["n_frames"] = len(set(df_verb["frame_name"]))
        abic_list += temp_df_abic.to_dict("records")
    return pd.DataFrame(abic_list)


def decide_params(vsc, df_abic, max_n_frames):
    min_diff_n, min_c = 1e10, -1
    for c in tqdm([i / 10 for i in range(51)]):
        _, df_output = vsc.aggregate_abic(df_abic, c, max_n_frames, max_n_frames)
        diff_n = abs(sum(df_output["n_frames"]) - sum(df_output["n_clusters"]))

        if min_diff_n >= diff_n:
            min_diff_n = diff_n
            min_c = c
    return {"diff_n": diff_n, "min_c": min_c}


def main():
    args = parse_args()
    path_dict = make_dir_path(
        args.input_path,
        args.dev_path,
        args.output_path,
        args.resource,
        args.model_name,
        args.criterion,
    )

    if args.layer != -1:
        layer = args.layer
    else:
        max_score, layer = -1, -1
        for path in glob.glob(path_dict["input_dev"] + "score_*.json"):
            with open(path, "r") as f:
                score = json.load(f)["score"]
            if max_score < score:
                max_score = score
                layer = int(path.split("/")[-1].split("_")[1].split(".json")[0])

    df = pd.read_json(
        path_dict["input"] + "exemplars.jsonl", orient="records", lines=True
    )
    vec_array = np.load(
        path_dict["input"] + "vec_" + str(layer).zfill(2) + ".npz", allow_pickle=True
    )["vec"]

    max_n_frames_dev = max(
        [len(f) for f in df[df["sets"] == "dev"].groupby("verb").agg(set)["frame_name"]]
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
            params = decide_params(vsc, df_abic, max_n_frames_dev)
            file_path = "params_" + str(args.layer).zfill(2) + ".json"
            with open(path_dict["output_dev"] + file_path, "w") as f:
                json.dump(params, f, indent=2, ensure_ascii=False)

    elif args.sets == "test":
        max_n_frames = max([len(f) for f in df.groupby("verb").agg(set)["frame_name"]])
        max_n_clusters = max_n_frames_dev
        if args.criterion == "abic":
            file_path = "params_" + str(args.layer).zfill(2) + ".json"
            with open(path_dict["output_dev"] + file_path, "r") as f:
                params = json.load(f)

    df_cm, df_output = vsc.aggregate_abic(
        df_abic, params["min_c"], max_n_frames, max_n_clusters
    )
    score_dict = vsc.scoring(df_output)

    file_path = "confusion_matrix_" + str(args.layer).zfill(2) + ".jsonl"
    df_cm.to_json(
        path_dict["output_" + args.sets] + file_path,
        orient="records",
        force_ascii=False,
        lines=True,
    )

    file_path = "n_frames_clusters_" + str(args.layer).zfill(2) + ".jsonl"
    df_output.to_json(
        path_dict["output_" + args.sets] + file_path,
        orient="records",
        force_ascii=False,
        lines=True,
    )

    file_path = "score_" + str(args.layer).zfill(2) + ".json"
    with open(path_dict["output_" + args.sets] + file_path, "w") as f:
        json.dump(score_dict, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
