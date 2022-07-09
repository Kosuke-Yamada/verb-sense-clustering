# -*-coding:utf-8-*-

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from _verb_sense_clustering import VerbSenseClustering


def parse_args():
    RESOURCE = ["framenet", "propbank"][0]
    MODEL_NAME = [
        "elmo",
        "bert-base-uncased",
        "bert-large-uncased",
        "albert-base-v2",
        "roberta-base",
        "gpt2",
        "xlnet-base-cased",
    ][0]
    LAYER = 0
    SETS = ["dev", "test"][0]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="../data/experiment_frame_distinction/embeddings",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../data/experiment_frame_distinction/frame_distinction",
    )

    parser.add_argument("--resource", type=str, default=RESOURCE)
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--covariance_type", type=str, default="spherical")
    parser.add_argument("--sets", type=str, default=SETS)
    return parser.parse_args()


def make_dir_path(input_path, output_path, resource, model_name):
    path_dict = {
        "input": "/".join([input_path, resource, model_name]),
        "output_dev": "/".join([output_path, resource, model_name, "dev"]),
        "output_test": "/".join([output_path, resource, model_name, "test"]),
    }
    for key, path in path_dict.items():
        path_dict[key] = os.path.abspath(path) + "/"
        if "output" in key:
            if not os.path.isdir(path_dict[key]):
                os.makedirs(path_dict[key])
    return path_dict


def main():
    args = parse_args()
    path_dict = make_dir_path(
        args.input_path,
        args.output_path,
        args.resource,
        args.model_name,
    )

    if args.layer != -1:
        layer = args.layer
    else:
        max_score, layer = -1, -1
        for path in glob.glob(path_dict["output_dev"] + "score_*.json"):
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

    df = df[df["sets"] == args.sets]
    vec_array = vec_array[df.index]
    df = df.reset_index()

    vsc = VerbSenseClustering(args.n_seeds, args.covariance_type)

    output_list = []
    for verb in tqdm(sorted(set(df["verb"]))):
        df_verb = df[df["verb"] == verb].copy()
        vec_verb = vec_array[df_verb.index]
        df_verb["cluster_id"] = vsc.run_clustering(
            vec_verb, len(set(df_verb["frame_name"]))
        )
        score = vsc.calc_matching_scores(df_verb["frame_name"], df_verb["cluster_id"])
        output_list.append({"verb": verb, "score": score})

    df_output = pd.DataFrame(output_list)
    df_output.to_json(
        path_dict["output_" + args.sets]
        + "verb_scores_"
        + str(layer).zfill(2)
        + ".jsonl",
        orient="records",
        force_ascii=False,
        lines=True,
    )

    with open(
        path_dict["output_" + args.sets] + "score_" + str(layer).zfill(2) + ".json", "w"
    ) as f:
        json.dump({"score": df_output["score"].mean()}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
