# -*-coding:utf-8-*-

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
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
    ][3]
    LAYER = -1
    SETS = ["dev", "test"][1]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="../data/experiment_frame_distinction/embeddings",
    )
    parser.add_argument(
        "--dev_path",
        type=str,
        default="../data/experiment_frame_distinction/frame_distinction",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../data/experiment_frame_distinction/visualization",
    )

    parser.add_argument("--resource", type=str, default=RESOURCE)
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--covariance_type", type=str, default="spherical")
    parser.add_argument("--sets", type=str, default=SETS)

    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--verb", type=str, default="")
    return parser.parse_args()


def make_dir_path(input_path, dev_path, output_path, resource, model_name, sets):
    path_dict = {
        "input": "/".join([input_path, resource, model_name]),
        "dev": "/".join([dev_path, resource, model_name, "dev"]),
        "output": "/".join([output_path, resource, model_name, sets]),
    }
    for key, path in path_dict.items():
        path_dict[key] = os.path.abspath(path) + "/"
        if "output" in key:
            if not os.path.isdir(path_dict[key]):
                os.makedirs(path_dict[key])
    return path_dict


def project_embeddings(df, vec_array, random_state):
    df = df.copy()
    if len(vec_array) >= 2:
        df.loc[:, ["x", "y"]] = TSNE(
            n_components=2, random_state=random_state
        ).fit_transform(vec_array)
    else:
        df.loc[:, ["x", "y"]] = np.array([0, 0])
    return df


def save_visualization(output_path, df, title, detail=True):
    fig = plt.figure(figsize=(8, 8))
    fig.set_facecolor("white")

    grid = sns.scatterplot(
        x="x",
        y="y",
        data=df,
        hue="frame_name",
        style="frame_name",
        s=200,
        edgecolor="white",
        legend=detail,
    )
    grid.set(xlabel="", ylabel="")

    if detail == True:
        handles, labels = grid.get_legend_handles_labels()
        plt.legend(handles, labels, fontsize=20)
        plt.title(title, fontsize=24)

        for (x, y, t) in zip(df["x"], df["y"], df["cluster_id"]):
            plt.text(x, y, str(t), ha="center", va="bottom", color="black", fontsize=18)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    args = parse_args()
    path_dict = make_dir_path(
        args.input_path,
        args.dev_path,
        args.output_path,
        args.resource,
        args.model_name,
        args.sets,
    )

    if args.layer != -1:
        layer = args.layer
    else:
        max_score, layer = -1, -1
        for path in glob.glob(path_dict["dev"] + "score_*.json"):
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

    if args.verb == "":
        verb_list = sorted(set(df["verb"]))
    else:
        verb_list = [args.verb]

    for verb in tqdm(verb_list):
        df_verb = df[df["verb"] == verb].copy()
        vec_verb = vec_array[df_verb.index]
        if args.model_name == "all-in-one-cluster":
            df_verb["cluster_id"] = vsc.run_all_in_one_cluster(vec_verb)
        else:
            df_verb["cluster_id"] = vsc.run_clustering(
                vec_verb, len(set(df_verb["frame_name"]))
            )

        df_prj = project_embeddings(df_verb, vec_verb, args.random_state)

        file_path = "-".join([verb, str(args.random_state).zfill(2), "w"]) + ".png"
        save_visualization(path_dict["output"] + file_path, df_prj, verb, True)

        file_path = "-".join([verb, str(args.random_state).zfill(2), "wo"]) + ".png"
        save_visualization(path_dict["output"] + file_path, df_prj, verb, False)


if __name__ == "__main__":
    main()
