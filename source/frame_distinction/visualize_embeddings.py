import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from tqdm import tqdm

from vsc.clustering import VerbSenseClustering
from vsc.data_utils import read_jsonl


def project_embeddings(df, vec_array, random_state):
    df = df.copy()
    if len(vec_array) >= 2:
        df.loc[:, ["x", "y"]] = TSNE(
            n_components=2, random_state=random_state
        ).fit_transform(vec_array)
    else:
        df.loc[:, ["x", "y"]] = np.array([0, 0])
    return df


def save_visualization(output_file, df, title, detail=True):
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

        for x, y, t in zip(df["x"], df["y"], df["cluster_id"]):
            plt.text(
                x,
                y,
                str(t),
                ha="center",
                va="bottom",
                color="black",
                fontsize=18,
            )
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.layer == "best":
        best_score, layer = -1, -1
        for file in args.input_dev_dir.glob("score-*.json"):
            with open(file, "r") as f:
                score = json.load(f)["score"]
            if best_score < score:
                best_score = score
                layer = file.name.split("-")[1].split(".json")[0]
    else:
        layer = args.layer

    df = pd.DataFrame(read_jsonl(args.input_dir / "exemplars.jsonl"))
    vec_array = np.load(args.input_dir / f"vec-{layer}.npz", allow_pickle=True)[
        "vec"
    ]

    df = df[df["sets"] == args.sets]
    vec_array = vec_array[df.index]
    df = df.reset_index()

    vsc = VerbSenseClustering(args.n_seeds, args.covariance_type)

    if args.verb == "all_verbs":
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

        file = "-".join([verb, str(args.random_state).zfill(2), "w"]) + ".png"
        save_visualization(args.output_dir / file, df_prj, verb, True)

        file = "-".join([verb, str(args.random_state).zfill(2), "wo"]) + ".png"
        save_visualization(args.output_dir / file, df_prj, verb, False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--input_dev_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument(
        "--resource", type=str, choices=["framenet", "propbank"]
    )
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
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--covariance_type", type=str, default="spherical")
    parser.add_argument("--sets", type=str, choices=["dev", "test"])

    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--verb", type=str, default="all_verbs")
    args = parser.parse_args()
    print(args)
    main(args)
